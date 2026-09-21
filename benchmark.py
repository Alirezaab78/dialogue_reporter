"""مقایسه زمان رونویسی Faster-Whisper روی CPU و CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path

from transcriber import Transcriber


def run_once(audio_path: Path, device: str, model_size: str) -> float:
    transcriber = Transcriber(model_size=model_size, device=device)  # type: ignore[arg-type]
    # دانلود/بارگذاری مدل خارج از زمان decode انجام می‌شود تا مقایسه منصفانه‌تر باشد.
    transcriber.load_model()
    result = transcriber.transcribe(audio_path)
    print(f"{result.hardware.upper()}: {result.processing_time_seconds:.3f} ثانیه")
    return result.processing_time_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description="مقایسه CPU و CUDA در Faster-Whisper")
    parser.add_argument("audio", nargs="?", default="output_clean.wav", help="مسیر فایل صوتی")
    parser.add_argument("--model", default="base", help="اندازه مدل، مثل base یا small")
    args = parser.parse_args()
    audio_path = Path(args.audio)
    if not audio_path.is_file():
        parser.error(f"فایل صوتی پیدا نشد: {audio_path}")

    cpu_time = run_once(audio_path, "cpu", args.model)
    try:
        gpu_time = run_once(audio_path, "cuda", args.model)
    except (RuntimeError, ImportError) as exc:
        print(f"CUDA در دسترس نیست یا مدل روی GPU load نشد؛ مقایسه GPU انجام نشد: {exc}")
        return 0

    if gpu_time > 0:
        print(f"نسبت سرعت CPU/GPU: {cpu_time / gpu_time:.2f}x")
        print(f"کاهش زمان پردازش با GPU: {(1 - gpu_time / cpu_time) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
