"""تبدیل گفتار به متن با Faster-Whisper، با پشتیبانی CPU و CUDA."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

try:
    from faster_whisper import WhisperModel
except ImportError:  # برای امکان اجرای تست‌های واحد بدون نصب مدل/کتابخانه
    WhisperModel = None  # type: ignore[assignment,misc]


Device = Literal["cpu", "cuda", "auto"]


@dataclass(frozen=True)
class TranscriptionSegment:
    """یک قطعه زمانی از خروجی رونویسی."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    """خروجی ساختاریافته و قابل تبدیل به JSON."""

    processing_time_seconds: float
    hardware: Literal["cpu", "cuda"]
    segments: list[TranscriptionSegment]
    full_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Transcriber:
    """رونویسی صوت با انتخاب خودکار یا صریح سخت‌افزار."""

    def __init__(
        self,
        model_size: str = "base",
        device: Device = "auto",
        compute_type: str | None = None,
        model_path: str | None = None,
        language: str | None = None,
    ) -> None:
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device باید یکی از cpu، cuda یا auto باشد.")
        if not model_size:
            raise ValueError("model_size نمی‌تواند خالی باشد.")

        self.model_size = model_size
        self.requested_device = device
        self.device = self._resolve_device(device)
        self.compute_type = compute_type or ("float16" if self.device == "cuda" else "int8")
        self.model_path = model_path
        self.language = language
        self._model: Any | None = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import ctranslate2

            return ctranslate2.get_cuda_device_count() > 0
        except (ImportError, AttributeError, RuntimeError, OSError):
            return False

    @classmethod
    def _resolve_device(cls, device: Device) -> Literal["cpu", "cuda"]:
        if device == "cuda":
            return "cuda"
        if device == "auto":
            return "cuda" if cls._cuda_available() else "cpu"
        return "cpu"

    def load_model(self) -> Any:
        """مدل را یک‌بار load می‌کند؛ این مرحله را در benchmark از زمان decode جدا می‌کنیم."""
        if self._model is not None:
            return self._model
        if WhisperModel is None:
            raise ImportError(
                "faster-whisper نصب نیست. ابتدا با `python -m pip install -r requirements.txt` نصب کنید."
            )

        try:
            model_name = self.model_path or self.model_size
            self._model = WhisperModel(
                model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            if self.device == "cuda":
                raise RuntimeError(
                    "بارگذاری مدل روی CUDA شکست خورد؛ نصب درایور NVIDIA، CUDA/cuDNN و PATH را بررسی کنید."
                ) from exc
            raise RuntimeError(f"بارگذاری مدل Faster-Whisper شکست خورد: {exc}") from exc
        return self._model

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """فایل صوتی را رونویسی و زمان، قطعات زمانی و متن کامل را برمی‌گرداند."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"فایل صوتی پیدا نشد: {path}")
        if not path.is_file():
            raise ValueError(f"مسیر ورودی فایل نیست: {path}")

        model = self.load_model()
        options: dict[str, Any] = {"vad_filter": True}
        if self.language:
            options["language"] = self.language

        started = time.perf_counter()
        try:
            segments, _info = model.transcribe(str(path), **options)
            structured_segments = [
                TranscriptionSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(segment.text).strip(),
                )
                for segment in segments
            ]
        except Exception as exc:
            raise RuntimeError(f"رونویسی فایل صوتی شکست خورد: {exc}") from exc

        elapsed = time.perf_counter() - started
        full_text = " ".join(item.text for item in structured_segments if item.text).strip()
        return TranscriptionResult(
            processing_time_seconds=elapsed,
            hardware=self.device,
            segments=structured_segments,
            full_text=full_text,
        )
