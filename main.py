"""اجرای کامل pipeline: صوت -> متن -> گزارش پزشکی."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from medical_reporter import MedicalReporter
from audio_preprocessor import AudioPreprocessor
from audio_recorder import AudioRecorder
from transcriber import Transcriber


def _configure_utf8_console() -> None:
    """در ویندوز، کنسول را برای نمایش امن متن فارسی آماده می‌کند."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run_pipeline(
    audio_path: str | Path,
    report_output_path: str | Path = "final_medical_report.txt",
    transcript_output_path: str | Path = "result.txt",
    model_size: str = "base",
    device: str = "auto",
    language: str | None = "fa",
    record: bool = False,
    record_output_path: str | Path = "recorded_session.wav",
    preprocessed_output_path: str | Path = "preprocessed_audio.wav",
    recorder: Any | None = None,
    preprocessor: Any | None = None,
    transcriber: Any | None = None,
    reporter: Any | None = None,
) -> tuple[Path, float]:
    """کل pipeline را اجرا و (مسیر گزارش، زمان کل) را برمی‌گرداند."""
    started = time.perf_counter()
    if record or audio_path is None:
        raw_audio_path = (recorder or AudioRecorder()).record(record_output_path)
    else:
        raw_audio_path = Path(audio_path)

    # نقطه اتصال مستقل برای فاز بعدی: diarization می‌تواند بعد از این مرحله
    # و پیش از Transcriber روی فایل preprocessed_audio اعمال شود.
    processed_audio_path = (preprocessor or AudioPreprocessor()).preprocess(
        raw_audio_path,
        preprocessed_output_path,
    )

    active_transcriber = transcriber or Transcriber(
        model_size=model_size,
        device=device,  # type: ignore[arg-type]
        language=language,
    )
    transcription = active_transcriber.transcribe(processed_audio_path)

    transcript_path = Path(transcript_output_path)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(transcription.full_text, encoding="utf-8", newline="\n")

    report = (reporter or MedicalReporter()).generate_report(transcription.full_text)
    report_path = Path(report_output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")

    elapsed = time.perf_counter() - started
    return report_path, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio -> Faster-Whisper -> Medical Report")
    parser.add_argument("audio", nargs="?", default=None, help="مسیر فایل صوتی موجود")
    parser.add_argument("--record", action="store_true", help="ضبط زنده از میکروفون")
    parser.add_argument("--record-output", default="recorded_session.wav", help="فایل WAV ضبط خام")
    parser.add_argument("--preprocessed-output", default="preprocessed_audio.wav", help="فایل WAV پس از پیش‌پردازش")
    parser.add_argument("-o", "--output", default="final_medical_report.txt", help="فایل گزارش نهایی")
    parser.add_argument("--transcript-output", default="result.txt", help="فایل transcript میانی")
    parser.add_argument("--model", default="base", help="اندازه مدل Faster-Whisper")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--language", default="fa", help="کد زبان صوت؛ برای تشخیص خودکار خالی بگذارید")
    args = parser.parse_args()

    try:
        output, elapsed = run_pipeline(
            audio_path=args.audio,
            record=args.record,
            record_output_path=args.record_output,
            preprocessed_output_path=args.preprocessed_output,
            report_output_path=args.output,
            transcript_output_path=args.transcript_output,
            model_size=args.model,
            device=args.device,
            language=args.language or None,
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        print(f"خطا در اجرای pipeline: {exc}", file=sys.stderr)
        return 1

    print(f"pipeline با موفقیت تمام شد؛ گزارش: {output}; زمان کل: {elapsed:.2f} ثانیه")
    return 0


if __name__ == "__main__":
    _configure_utf8_console()
    raise SystemExit(main())
