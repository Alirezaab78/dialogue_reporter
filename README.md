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

## گام دوم: رونویسی با Faster-Whisper

وابستگی `faster-whisper` در `requirements.txt` اضافه شده است. برای نصب:

```bash
python -m pip install -r requirements.txt
```

در ویندوز، اجرای CUDA به درایور NVIDIA و DLLهای سازگار CUDA/cuDNN نیاز دارد. نسخه درایور را از NVIDIA و نسخه CUDA/cuDNN سازگار با CTranslate2 را نصب کنید و مسیر DLLها را در `PATH` قرار دهید. اگر CUDA قابل بارگذاری نباشد، `device="auto"` به CPU برمی‌گردد.

اجرای رونویسی مستقیم:

```bash
python -c "from transcriber import Transcriber; print(Transcriber(model_size='base', device='auto').transcribe('output_clean.wav').to_dict())"
```

بنچمارک CPU و GPU:

```bash
python benchmark.py output_clean.wav --model base
```

برای تست اولیه می‌توان مدل `small` را نیز استفاده کرد؛ مدل‌ها در اولین اجرا دانلود می‌شوند. تست‌های واحد گام دوم با مدل واقعی کار نمی‌کنند و برای سرعت، API مدل را mock می‌کنند:

```bash
python -m pytest -q
```
