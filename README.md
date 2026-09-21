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

## گام سوم: تولید گزارش پزشکی

برای استفاده از [medical_reporter.py](medical_reporter.py)، فایل `.env.example` را به `.env` کپی و مقادیر آن را تنظیم کنید:

```bash
copy .env.example .env
```

برای OpenAI رسمی، مقدار `OPENAI_BASE_URL` همان `https://api.openai.com/v1` است. برای سرویس‌های OpenAI-compatible فقط `OPENAI_BASE_URL` و `OPENAI_API_KEY` را تغییر دهید؛ نیازی به تغییر کد نیست.

اجرای تست ساده:

```bash
python medical_reporter.py
```

استفاده در کد:

```python
from medical_reporter import MedicalReporter

report = MedicalReporter().generate_report(transcript)
```

لاگ فقط latency و تعداد توکن‌های ورودی/خروجی را ثبت می‌کند و transcript یا گزارش پزشکی را چاپ نمی‌کند.

## اجرای End-to-End Pipeline

اجرای کامل صوت تا گزارش نهایی:

```bash
python main.py output_clean.wav --model base --device auto
```

این دستور transcript را در `result.txt` و گزارش نهایی را با UTF-8 در `final_medical_report.txt` ذخیره می‌کند و فقط مسیر خروجی و زمان کل را در ترمینال نمایش می‌دهد. مسیرها قابل تغییرند:

```bash
python main.py sample.wav -o reports/final_medical_report.txt --transcript-output reports/result.txt --device cuda
```

برای تبدیل یک transcript موجود به گزارش مستقل:

```bash
python medical_reporter.py result.txt -o report.txt
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
