# dictation-local-windows

Голосовой ввод для Windows: Alt+Space → говорите → отпустите → текст вставится в активное окно.

## Что внутри

- Два бэкенда на выбор через `MODEL_KIND` в `diktovka.py`:
  - `"whisper"` — `OpenVINO/whisper-medium-fp16-ov` через `openvino_genai.WhisperPipeline`, фоллбэк NPU → GPU → CPU.
  - `"gigaam"` — `Andrewsab/gigaam-v3-e2e-rnnt-ov` (GigaAM v3 e2e RNN-T, русский, с пунктуацией): три OpenVINO IR-подмодели (encoder/decoder/joint) + SentencePiece-токенизатор, log-mel считается torchaudio'м (htk, 64 бина, n_fft=320, win=320, hop=160, center=False), greedy RNN-T декодинг. Рекомендованное устройство — `GPU.0` (iGPU), фоллбэк CPU. NPU не поддерживается (encoder/joint падают на dynamic shapes). Удивительно, что на CPU работает быстрее, чем на iGPU.
- Модели подтягиваются готовыми через `huggingface_hub.snapshot_download` — никакой локальной конвертации.
- Захват — `sounddevice` (16 кГц, mono, float32, чанки по 512 сэмплов).
- Хоткей: `keyboard.add_hotkey("alt+space", ..., suppress=True)` — старт по нажатию (гасит системное меню), `on_release_key` для `alt` и `space` — стоп по отпусканию любой из них.
- Вставка через буфер обмена + `Shift+Insert` (PuTTY / Windows Terminal / cmd / GUI), активное окно запоминается до начала записи и восстанавливается перед вставкой.

## Установка

Базовое:

```
pip install sounddevice numpy keyboard pyperclip pywin32 openvino openvino-genai huggingface_hub
```

Для GigaAM дополнительно:

```
pip install torch torchaudio sentencepiece
```

## Запуск

```
python diktovka.py
```

В файле в начале выбирается бэкенд: `MODEL_KIND = "whisper"` или `"gigaam"`. Первый запуск скачает модель в кэш HuggingFace (`~/.cache/huggingface/hub/`), дальше — мгновенный старт.
