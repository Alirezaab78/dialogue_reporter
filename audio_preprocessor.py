"""پیش‌پردازش فایل صوتی برای آماده‌سازی ورودی Whisper.

این ماژول فقط مسئول آماده‌سازی صوت است و هیچ کاری با رونویسی یا تشخیص
هویت گوینده انجام نمی‌دهد.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import noisereduce as nr
import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from scipy.io import wavfile


PathLike = Union[str, Path]


class AudioPreprocessor:
    """تمیزسازی و استانداردسازی صوت برای مدل‌های تشخیص گفتار."""

    SUPPORTED_EXTENSIONS = {".wav", ".mp3"}

    def __init__(
        self,
        noise_reduction_strength: float = 0.75,
        target_rms_dbfs: float = -20.0,
        max_gain_db: float = 12.0,
        peak_limit_dbfs: float = -1.0,
    ) -> None:
        if not 0.0 <= noise_reduction_strength <= 1.0:
            raise ValueError("noise_reduction_strength باید بین ۰ و ۱ باشد.")
        if max_gain_db < 0:
            raise ValueError("max_gain_db نمی‌تواند منفی باشد.")

        self.noise_reduction_strength = noise_reduction_strength
        self.target_rms_dbfs = target_rms_dbfs
        self.max_gain_db = max_gain_db
        self.peak_limit_dbfs = peak_limit_dbfs

    def preprocess(self, input_path: PathLike, output_path: PathLike | None = None) -> Path:
        """فایل WAV/MP3 را پردازش و به WAV مونو با نرخ ۱۶ کیلوهرتز تبدیل می‌کند."""
        source = self._validate_input_path(input_path)
        destination = self._make_output_path(source, output_path)

        try:
            # ابتدا به مشخصات استاندارد تبدیل می‌کنیم تا تمام مراحل روی یک سیگنال ثابت انجام شوند.
            audio = (
                AudioSegment.from_file(source)
                .set_channels(1)
                .set_frame_rate(16_000)
                .set_sample_width(2)
            )
        except (CouldntDecodeError, OSError, ValueError) as exc:
            raise ValueError(f"فایل صوتی قابل decode نیست: {source}") from exc

        samples = self._segment_to_float32(audio)
        if samples.size == 0:
            raise ValueError("فایل صوتی خالی است.")

        # کاهش نویز ایستا؛ شدت محافظه‌کارانه برای حفظ فرکانس‌های گفتاری انتخاب شده است.
        cleaned = nr.reduce_noise(
            y=samples,
            sr=16_000,
            stationary=True,
            prop_decrease=self.noise_reduction_strength,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
        )

        # فشرده‌سازی ملایم دامنه مانند AGC: بخش‌های بلند کمی مهار می‌شوند و سپس
        # کل سیگنال تا سطح RMS مناسب تقویت می‌شود؛ در نتیجه صدای آرام واضح‌تر می‌ماند.
        processed = self._automatic_gain_control(cleaned)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            # scipy برای نوشتن WAV به ffmpeg وابسته نیست؛ بنابراین پردازش WAV خام
            # حتی در سیستمی که ffmpeg نصب نشده است نیز انجام می‌شود.
            wavfile.write(destination, 16_000, self._float32_to_int16(processed))
        except OSError as exc:
            raise OSError(f"ذخیره فایل خروجی ممکن نشد: {destination}") from exc

        return destination

    @classmethod
    def _validate_input_path(cls, input_path: PathLike) -> Path:
        source = Path(input_path)
        if not source.exists():
            raise FileNotFoundError(f"فایل ورودی پیدا نشد: {source}")
        if not source.is_file():
            raise ValueError(f"مسیر ورودی فایل نیست: {source}")
        if source.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError("فرمت ورودی باید WAV یا MP3 باشد.")
        return source

    @staticmethod
    def _make_output_path(source: Path, output_path: PathLike | None) -> Path:
        if output_path is not None:
            destination = Path(output_path)
            if destination.suffix.lower() != ".wav":
                raise ValueError("فرمت فایل خروجی باید WAV باشد.")
            return destination
        return source.with_name(f"{source.stem}_processed.wav")

    @staticmethod
    def _segment_to_float32(audio: AudioSegment) -> np.ndarray:
        samples = np.asarray(audio.get_array_of_samples(), dtype=np.float32)
        return samples / 32768.0

    def _automatic_gain_control(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        # RMS محلی با پنجره ۴۰۰ میلی‌ثانیه‌ای؛ این کار برخلاف نرمال‌سازی ساده،
        # بخش‌های آرام‌تر مکالمه (مثلاً صدای بیمار) را به‌طور مستقل تقویت می‌کند.
        window_size = 6_400
        kernel = np.ones(window_size, dtype=np.float32) / window_size
        local_power = np.convolve(np.square(samples), kernel, mode="same")
        local_rms = np.sqrt(local_power + 1e-12)
        target_rms = 10.0 ** (self.target_rms_dbfs / 20.0)
        gain = np.minimum(target_rms / np.maximum(local_rms, 1e-8), 10.0 ** (self.max_gain_db / 20.0))

        # صاف‌کردن تغییرات gain از شنیده‌شدن نوسان ناگهانی حجم صدا جلوگیری می‌کند.
        smooth_kernel = np.ones(801, dtype=np.float32) / 801
        gain = np.convolve(gain, smooth_kernel, mode="same")
        amplified = samples * gain

        # limiter نرم برای جلوگیری از clipping پس از تقویت.
        peak = float(np.max(np.abs(amplified)))
        allowed_peak = 10.0 ** (self.peak_limit_dbfs / 20.0)
        if peak > allowed_peak:
            amplified *= allowed_peak / peak
        return np.clip(amplified, -1.0, 1.0)

    @staticmethod
    def _float32_to_int16(samples: np.ndarray) -> np.ndarray:
        """سیگنال float را به PCM 16-bit مناسب WAV تبدیل می‌کند."""
        return np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="پیش‌پردازش فایل صوتی برای Whisper")
    parser.add_argument("input", help="مسیر فایل WAV یا MP3")
    parser.add_argument("-o", "--output", help="مسیر WAV خروجی")
    args = parser.parse_args()
    result = AudioPreprocessor().preprocess(args.input, args.output)
    print(f"فایل پردازش‌شده ذخیره شد: {result}")
