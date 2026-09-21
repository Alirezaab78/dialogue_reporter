"""تبدیل transcript مکالمه پزشکی به گزارش رسمی فارسی با OpenAI-compatible API."""

from __future__ import annotations

import logging
import time
import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # اجازه می‌دهد تست‌های ساختاری بدون نصب وابستگی اجرا شوند.
    def load_dotenv() -> bool:
        return False

try:
    from openai import OpenAI
except ImportError:  # در زمان اجرا با پیام واضح کنترل می‌شود.
    OpenAI = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


class MedicalReporter:
    """کلاینت کوچک و مقاوم برای تولید گزارش پزشکی فارسی."""

    DEFAULT_MODEL = "gpt-5.6-terra"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_TIMEOUT_SECONDS = 30.0
    DEFAULT_MAX_RETRIES = 2

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: Any | None = None,
    ) -> None:
        load_dotenv()
        import os

        self.model = model or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", self.DEFAULT_BASE_URL)
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.model:
            raise ValueError("نام مدل نمی‌تواند خالی باشد.")
        if timeout <= 0:
            raise ValueError("timeout باید بزرگ‌تر از صفر باشد.")
        if max_retries < 0:
            raise ValueError("max_retries نمی‌تواند منفی باشد.")

        self.client = client or self._build_client()

    def _build_client(self) -> Any:
        if OpenAI is None:
            raise ImportError("کتابخانه openai نصب نیست؛ `python -m pip install -r requirements.txt` را اجرا کنید.")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY در فایل .env یا متغیرهای محیطی تنظیم نشده است.")
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def generate_report(self, transcript: str) -> str:
        """transcript را به گزارش رسمی فارسی تبدیل می‌کند.

        متن transcript و گزارش هرگز در لاگ نوشته نمی‌شوند.
        """
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("transcript باید یک رشته غیرخالی باشد.")

        messages = [
            {
                "role": "system",
                "content": (
                    "شما یک دستیار مستندسازی پزشکی هستید. مکالمه زیر را فقط به یک گزارش رسمی و "
                    "ساختاریافته فارسی تبدیل کن. تشخیص جدید، حدس یا اطلاعاتی خارج از transcript اضافه نکن. "
                    "گزارش را با این بخش‌ها بنویس: شکایت اصلی، شرح حال و علائم، سوابق/داروها، "
                    "یافته‌های ذکرشده، ارزیابی، برنامه و نکات پیگیری. اگر داده‌ای موجود نیست، بنویس «ذکر نشده است»."
                ),
            },
            {"role": "user", "content": transcript.strip()},
        ]

        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
            )
        except Exception as exc:
            # عمداً متن خطا/درخواست را log نمی‌کنیم؛ ممکن است حاوی داده پزشکی باشد.
            logger.error("medical_report_request_failed error_type=%s", type(exc).__name__)
            raise RuntimeError("تولید گزارش پزشکی با خطا مواجه شد.") from exc

        latency = time.perf_counter() - started
        usage = getattr(response, "usage", None)
        prompt_tokens = self._usage_value(usage, "prompt_tokens")
        completion_tokens = self._usage_value(usage, "completion_tokens")
        logger.info(
            "medical_report_generated latency_seconds=%.3f prompt_tokens=%s completion_tokens=%s",
            latency,
            prompt_tokens,
            completion_tokens,
        )

        try:
            report = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("پاسخ API ساختار متنی معتبر نداشت.") from exc
        if not isinstance(report, str) or not report.strip():
            raise RuntimeError("مدل گزارش متنی خالی برگرداند.")
        return report.strip()

    @staticmethod
    def _usage_value(usage: Any, name: str) -> int:
        if usage is None:
            return 0
        value = usage.get(name, 0) if isinstance(usage, dict) else getattr(usage, name, 0)
        return int(value or 0)


def generate_report_from_file(
    input_path: str | Path = "result.txt",
    output_path: str | Path = "report.txt",
    reporter: MedicalReporter | None = None,
) -> Path:
    """متن UTF-8 را از فایل می‌خواند و گزارش UTF-8 تولیدشده را ذخیره می‌کند."""
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"فایل متن ورودی پیدا نشد: {source}")

    transcript = source.read_text(encoding="utf-8-sig")
    report = (reporter or MedicalReporter()).generate_report(transcript)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8", newline="\n")
    return destination


def _configure_utf8_console() -> None:
    """در ویندوز، کنسول را برای نمایش امن متن فارسی آماده می‌کند."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    _configure_utf8_console()
    parser = argparse.ArgumentParser(description="تبدیل transcript پزشکی UTF-8 به report.txt")
    parser.add_argument("input", nargs="?", default="result.txt", help="فایل transcript ورودی")
    parser.add_argument("-o", "--output", default="report.txt", help="فایل گزارش خروجی")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        output = generate_report_from_file(args.input, args.output)
        print(f"گزارش پزشکی در فایل زیر ذخیره شد: {output}")
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
