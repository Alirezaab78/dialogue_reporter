# Audio Preprocessor

این پروژه گام اول آماده‌سازی صوت برای Whisper را پیاده‌سازی می‌کند: کاهش نویز، تقویت/نرمال‌سازی، تبدیل به مونو و نرخ نمونه‌برداری ۱۶ کیلوهرتز و ذخیره به‌صورت WAV.

## نصب

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

برای خواندن MP3، نصب بودن `ffmpeg` روی سیستم و قرار داشتن آن در PATH لازم است. فایل‌های WAV بدون آن نیز قابل پردازش هستند.

## اجرا

```bash
python audio_preprocessor.py path/to/input.mp3 -o path/to/processed.wav
```

یا در کد:

```python
from audio_preprocessor import AudioPreprocessor

output = AudioPreprocessor().preprocess("input.wav", "processed.wav")
print(output)
```

## اجرای تست‌ها

```bash
python -m pytest -q
```
