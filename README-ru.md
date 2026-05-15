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

Так как все питоновские библиотеки для захвата звука с микрофона "конфликтовали с драйверами Intel SST" (Так, мне сказал Google), мне пришлось установить виртуальный аудиокабель https://vb-audio.com/Cable/. Чтобы виртуальный кабель заработал, нужно сделать настройку в оснастке управления звуком Windows. Запустить ее можно нажав Win+R mmsys.cpl. В оснастке управления звуком я выбрал встороенный в ноутбук микрофон и перешёл в его свойства <img width="394" height="450" alt="image" src="https://github.com/user-attachments/assets/4be683be-48cb-4f0a-93b5-895243f45d5d" />
 На вкладке Listen я выбрал устройство виртуального кабеля, которое установил ранее <img width="405" height="458" alt="image" src="https://github.com/user-attachments/assets/55c95f1b-ca83-42c0-8b31-7a93244dc0df" />

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
