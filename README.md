# dictation-local-windows
Local dictation Windows on Whisper and Giga AM models.

Voice typing for Windows: Press and hold Alt+Space → speak → release → the text is pasted into the active window.

## What's inside

- Two backends to choose from via `MODEL_KIND` in `diktovka.py`:
  - `"whisper"` — `OpenVINO/whisper-medium-fp16-ov` via `openvino_genai.WhisperPipeline`, with an NPU → GPU → CPU fallback.
  - `"gigaam"` — `Andrewsab/gigaam-v3-e2e-rnnt-ov` (GigaAM v3 e2e RNN-T, Russian, with punctuation): consists of three OpenVINO IR sub-models (encoder/decoder/joint) + a SentencePiece tokenizer. Log-mel spectrograms are computed using `torchaudio` (htk, 64 bins, n_fft=320, win=320, hop=160, center=False), followed by greedy RNN-T decoding. Recommended device: `GPU.0` (iGPU), with a CPU fallback. NPU is not supported (the encoder/joint models fail on dynamic shapes). Surprisingly, for me it runs faster on the CPU than on the iGPU
- Models are downloaded ready-to-use via `huggingface_hub.snapshot_download` — no local conversion required.
- Audio capture via `sounddevice` (16 kHz, mono, float32, 512-sample chunks).
- Hotkey handling: `keyboard.add_hotkey("alt+space", ..., suppress=True)` — starts recording on press (suppresses the Windows system menu), and uses `on_release_key` for `alt` and `space` to stop recording upon releasing either key.
- Text pasting via clipboard + `Shift+Insert` (works in PuTTY / Windows Terminal / cmd / GUI); the active window is memorized before recording starts and is restored right before pasting.

## Installation

Base requirements:

```
pip install sounddevice numpy keyboard pyperclip pywin32 openvino openvino-genai huggingface_hub
```

Additional requirements for GigaAM:

```
pip install torch torchaudio sentencepiece
```

## Usage

```
python diktovka.py
```

You can select the backend at the top of the file: `MODEL_KIND = "whisper"` or `"gigaam"`. The first run will download the model into the HuggingFace cache (`~/.cache/huggingface/hub/`), and subsequent runs will start instantly.