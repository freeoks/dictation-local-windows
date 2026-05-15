#!/usr/bin/env python3
"""
Голосовой ввод для Windows с транскрибацией на Intel NPU/GPU через OpenVINO.

Зажмите Alt+Space → говорите → отпустите — распознанный текст вставится в
активное окно (Shift+Insert, работает в PuTTY, Windows Terminal, cmd и GUI).

Установка (Whisper):
    pip install sounddevice numpy keyboard pyperclip pywin32
    pip install openvino openvino-genai huggingface_hub

Дополнительно для GigaAM:
    pip install torch torchaudio sentencepiece

Поддерживаются:
    * Whisper (FP16, NPU/GPU/CPU) — OpenVINO/whisper-medium-fp16-ov
    * GigaAM v3 e2e RNN-T (FP16, GPU/CPU) — Andrewsab/gigaam-v3-e2e-rnnt-ov
      (русский, с пунктуацией; NPU не работает из-за dynamic shapes)
"""

import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
import win32gui

from huggingface_hub import snapshot_download

# ── Настройки ─────────────────────────────────────────────────────────────────
MODEL_KIND  = "gigaam"                           # "whisper" | "gigaam"

# Whisper
WHISPER_MODEL_ID   = "OpenVINO/whisper-medium-fp16-ov"
WHISPER_DEVICE     = "NPU"                       # NPU | GPU | CPU
WHISPER_LANGUAGE   = "<|ru|>"                    # токен языка; None = авто
WHISPER_MAX_TOKENS = 224

# GigaAM v3 e2e RNN-T
GIGAAM_MODEL_ID = "Andrewsab/gigaam-v3-e2e-rnnt-ov"
GIGAAM_DEVICE   = "CPU" #"GPU.0"                        # GPU.0 (iGPU) | GPU | CPU; NPU не работает

SAMPLE_RATE = 16000                              # 16 кГц для обеих моделей
MIN_SEC     = 0.4                                # короче — пропустить
# ─────────────────────────────────────────────────────────────────────────────


# ── Whisper-бэкенд ───────────────────────────────────────────────────────────

class WhisperTranscriber:
    """OpenVINO-GenAI WhisperPipeline с фоллбэком NPU → GPU → CPU."""

    def __init__(self) -> None:
        import openvino_genai
        print(f"Получение модели {WHISPER_MODEL_ID}…")
        model_dir = Path(snapshot_download(repo_id=WHISPER_MODEL_ID))
        print(f"  путь: {model_dir}")

        print(f"Инициализация WhisperPipeline (предпочтительно {WHISPER_DEVICE})…")
        last_err: Exception | None = None
        for dev in (WHISPER_DEVICE, "GPU", "CPU"):
            try:
                self._pipe = openvino_genai.WhisperPipeline(str(model_dir), dev)
                print(f"  устройство: {dev}\n")
                return
            except Exception as e:
                print(f"  {dev}: {e}")
                last_err = e
        raise RuntimeError(f"WhisperPipeline не инициализировался: {last_err}")

    def transcribe(self, audio: np.ndarray) -> str:
        kwargs = {"max_new_tokens": WHISPER_MAX_TOKENS, "task": "transcribe"}
        if WHISPER_LANGUAGE:
            kwargs["language"] = WHISPER_LANGUAGE
        result = self._pipe.generate(audio, **kwargs)
        text = result.texts[0] if getattr(result, "texts", None) else str(result)
        return text.strip()


# ── GigaAM-бэкенд ────────────────────────────────────────────────────────────

class GigaAMTranscriber:
    """
    OpenVINO-инференс GigaAM v3 e2e RNN-T: три подмодели (encoder/decoder/joint) +
    SentencePiece-токенизатор. Препроцессинг (log-mel, htk, n_mels=64,
    n_fft=320, win=320, hop=160, center=False) — torchaudio.
    """

    PRED_HIDDEN = 320
    BLANK_ID    = 1024                            # num_classes - 1
    MAX_SYMBOLS = 10

    def __init__(self) -> None:
        import openvino as ov
        import torch
        import torchaudio
        from sentencepiece import SentencePieceProcessor

        self._torch = torch

        print(f"Получение модели {GIGAAM_MODEL_ID}…")
        model_dir = Path(snapshot_download(repo_id=GIGAAM_MODEL_ID))
        print(f"  путь: {model_dir}")

        print(f"Инициализация GigaAM (предпочтительно {GIGAAM_DEVICE})…")
        core = ov.Core()
        last_err: Exception | None = None
        for dev in (GIGAAM_DEVICE, "GPU", "CPU"):
            try:
                self._encoder = core.compile_model(str(model_dir / "v3_e2e_rnnt_encoder.xml"), dev)
                self._decoder = core.compile_model(str(model_dir / "v3_e2e_rnnt_decoder.xml"), dev)
                self._joint   = core.compile_model(str(model_dir / "v3_e2e_rnnt_joint.xml"),   dev)
                print(f"  устройство: {dev}\n")
                break
            except Exception as e:
                print(f"  {dev}: {e}")
                last_err = e
        else:
            raise RuntimeError(f"GigaAM IR не инициализировался: {last_err}")

        self._featurizer = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=64,
            win_length=320,
            hop_length=160,
            n_fft=320,
            center=False,
            mel_scale="htk",
            norm=None,
        )

        self._tokenizer = SentencePieceProcessor()
        self._tokenizer.load(str(model_dir / "tokenizer.model"))

    def _features(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        wav = torch.from_numpy(audio).float().unsqueeze(0)            # [1, N]
        mel = self._featurizer(wav)                                   # [1, 64, T]
        log_mel = torch.log(mel.clamp_(min=1e-9, max=1e9))
        feat = log_mel.numpy().astype(np.float32)
        length = np.array([feat.shape[-1]], dtype=np.int64)
        return feat, length

    def transcribe(self, audio: np.ndarray) -> str:
        feat, feat_len = self._features(audio)
        enc_out = self._encoder([feat, feat_len])
        encoded = enc_out[self._encoder.output(0)]                    # [1, 768, T_enc]
        enc_len = int(enc_out[self._encoder.output(1)][0])

        tokens: list[int] = []
        h = np.zeros((1, 1, self.PRED_HIDDEN), dtype=np.float32)
        c = np.zeros((1, 1, self.PRED_HIDDEN), dtype=np.float32)
        last_label = np.array([[self.BLANK_ID]], dtype=np.int64)      # blank → нулевой эмбеддинг

        dec_out_dec = self._decoder.output(0)
        dec_out_h   = self._decoder.output(1)
        dec_out_c   = self._decoder.output(2)
        joint_out   = self._joint.output(0)

        for t in range(enc_len):
            f = encoded[:, :, t:t + 1]                                # [1, 768, 1]
            new_symbols = 0
            while new_symbols < self.MAX_SYMBOLS:
                dec_res = self._decoder([last_label, h, c])
                dec   = dec_res[dec_out_dec]                          # [1, 1, 320]
                h_new = dec_res[dec_out_h]
                c_new = dec_res[dec_out_c]
                dec_for_joint = np.transpose(dec, (0, 2, 1))          # [1, 320, 1] (channel-first)
                logits = self._joint([f, dec_for_joint])[joint_out]   # [1, 1, 1, 1025]
                k = int(np.argmax(logits[0, 0, 0, :]))
                if k == self.BLANK_ID:
                    break
                tokens.append(k)
                h, c = h_new, c_new
                last_label = np.array([[k]], dtype=np.int64)
                new_symbols += 1

        return self._tokenizer.decode(tokens).strip()


# ── Фабрика ──────────────────────────────────────────────────────────────────

def build_transcriber():
    if MODEL_KIND == "whisper":
        return WhisperTranscriber()
    if MODEL_KIND == "gigaam":
        return GigaAMTranscriber()
    raise ValueError(f"Неизвестный MODEL_KIND={MODEL_KIND!r} (ожидается 'whisper' | 'gigaam')")


# ── Запись с микрофона (sounddevice) ──────────────────────────────────────────

class Recorder:
    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._active = False

    def start(self) -> None:
        self._chunks = []
        self._active = True
        mic = next((item.get('index') for item in sd.query_devices() if item.get('name') == 'CABLE Output (VB-Audio Point)'), None)
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            device=mic,
            dtype="float32",
            blocksize=512,
            callback=self._cb,
        )
        self._stream.start()

    def _cb(self, indata: np.ndarray, *_args) -> None:
        if self._active:
            self._chunks.append(indata[:, 0].copy())

    def stop(self) -> np.ndarray:
        self._active = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._chunks:
            return np.zeros(0, dtype="float32")
        return np.concatenate(self._chunks)


# ── Вставка текста в активное окно ────────────────────────────────────────────

def paste_text(hwnd: int, text: str) -> None:
    """
    Кладёт текст в буфер обмена и шлёт Shift+Insert активному окну.
    Shift+Insert — универсальный аккорд вставки: работает в PuTTY,
    Windows Terminal, cmd, PowerShell, большинстве GUI-приложений.
    """
    try:
        backup = pyperclip.paste()
    except Exception:
        backup = ""

    pyperclip.copy(text)

    try:
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.08)
    except Exception:
        pass

    keyboard.send("shift+insert")
    time.sleep(0.15)

    try:
        pyperclip.copy(backup)
    except Exception:
        pass


# ── Обработка хоткея Alt+Space и главный цикл ─────────────────────────────────

def main() -> None:
    transcriber = build_transcriber()
    recorder    = Recorder()
    lock        = threading.Lock()
    state: dict = {"rec": False, "hwnd": 0}

    def on_start() -> None:
        with lock:
            if state["rec"]:
                return
            state["rec"]  = True
            state["hwnd"] = win32gui.GetForegroundWindow()
        recorder.start()
        print("● запись…", flush=True)

    def on_stop() -> None:
        with lock:
            if not state["rec"]:
                return
            state["rec"] = False

        audio = recorder.stop()
        hwnd  = state["hwnd"]
        print("■ обработка…", flush=True)

        def work() -> None:
            if len(audio) < SAMPLE_RATE * MIN_SEC:
                print("  (слишком коротко, пропуск)\n", flush=True)
                return
            try:
                text = transcriber.transcribe(audio)
            except Exception as e:
                print(f"  ошибка транскрибации: {e}\n", flush=True)
                return
            if text:
                print(f"  → {text}\n", flush=True)
                paste_text(hwnd, text)
            else:
                print("  (тишина)\n", flush=True)

        threading.Thread(target=work, daemon=True).start()

    # suppress=True гасит штатное системное меню окна, которое открывает Alt+Space
    keyboard.add_hotkey("alt+space", on_start, suppress=True)
    keyboard.on_release_key("alt",   lambda _e: on_stop())
    keyboard.on_release_key("space", lambda _e: on_stop())

    print("Готово. Зажмите Alt+Space — говорите — отпустите. Ctrl+C для выхода.\n")
    try:
        keyboard.wait()
    except KeyboardInterrupt:
        print("\nВыход.")


if __name__ == "__main__":
    main()
