"""ضبط صوت زنده از میکروفون به‌صورت WAV مونو، 16kHz و PCM 16-bit."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

try:
    import sounddevice as sd
except ImportError:  # پیام خطا در زمان استفاده کنترل می‌شود تا تست‌ها قابل اجرا بمانند.
    sd = None  # type: ignore[assignment]


class AudioRecordingError(RuntimeError):
    """خطای قابل‌فهم برای نبود میکروفون یا شکست ضبط."""


class AudioRecorder:
    """ضبط‌کننده پرتابل مبتنی بر sounddevice و PortAudio."""

    def __init__(self, sample_rate: int = 16_000, channels: int = 1, device: int | str | None = None) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate باید بزرگ‌تر از صفر باشد.")
        if channels != 1:
            raise ValueError("برای این pipeline فقط ضبط Mono پشتیبانی می‌شود.")
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device

    def default_input_device(self) -> int | str | None:
        """میکروفون ورودی پیش‌فرض سیستم را برمی‌گرداند."""
        self._require_sounddevice()
        if self.device is not None:
            return self.device
        try:
            default_device = sd.default.device[0]
            if default_device is None or default_device < 0:
                sd.query_devices(kind="input")
                return None  # sounddevice در این حالت دستگاه ورودی پیش‌فرض را انتخاب می‌کند.
            return default_device
        except Exception as exc:
            raise AudioRecordingError("میکروفون ورودی پیش‌فرض پیدا نشد یا قابل دسترسی نیست.") from exc

    def record(self, output_path: str | Path, duration_seconds: float | None = None) -> Path:
        """ضبط زمان‌دار یا تعاملی و ذخیره امن فایل WAV."""
        destination = self._validate_output_path(output_path)
        if duration_seconds is not None:
            if duration_seconds <= 0:
                raise ValueError("duration_seconds باید بزرگ‌تر از صفر باشد.")
            samples = self._record_for_duration(duration_seconds)
        else:
            self.default_input_device()
            samples = self._record_until_enter()
        return self._save_wav(samples, destination)

    def _record_for_duration(self, duration_seconds: float) -> np.ndarray:
        self._require_sounddevice()
        frames = int(round(duration_seconds * self.sample_rate))
        try:
            samples = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                device=self.default_input_device(),
            )
            sd.wait()
            return self._as_mono_int16(samples)
        except Exception as exc:
            raise AudioRecordingError("ضبط زمان‌دار از میکروفون شکست خورد.") from exc

    def _record_until_enter(self) -> np.ndarray:
        """با Enter شروع و با Enter دوم پایان می‌دهد."""
        self._require_sounddevice()
        input("برای شروع ضبط Enter را بزنید...")
        chunks: list[np.ndarray] = []
        lock = threading.Lock()

        def callback(indata: np.ndarray, _frames: int, _time: Any, status: Any) -> None:
            if status:
                # وضعیت فقط برای تشخیص خطاست؛ محتوای صوتی وارد log/console نمی‌شود.
                return
            with lock:
                chunks.append(np.array(indata, dtype=np.int16, copy=True))

        stream = None
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                device=self.default_input_device(),
                callback=callback,
            )
            stream.start()
            input("در حال ضبط... برای پایان Enter را بزنید.")
        except Exception as exc:
            raise AudioRecordingError("ضبط تعاملی از میکروفون شکست خورد.") from exc
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

        with lock:
            if not chunks:
                raise AudioRecordingError("هیچ نمونه صوتی از میکروفون دریافت نشد.")
            return self._as_mono_int16(np.concatenate(chunks, axis=0))

    def _save_wav(self, samples: np.ndarray, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
        try:
            wavfile.write(temporary, self.sample_rate, self._as_mono_int16(samples))
            temporary.replace(destination)
        except (OSError, ValueError) as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise AudioRecordingError(f"ذخیره فایل ضبط‌شده ممکن نشد: {destination}") from exc
        return destination

    @staticmethod
    def _as_mono_int16(samples: np.ndarray) -> np.ndarray:
        array = np.asarray(samples)
        if array.ndim == 2:
            array = array[:, 0]
        return np.clip(array, -32768, 32767).astype(np.int16, copy=False)

    @staticmethod
    def _validate_output_path(output_path: str | Path) -> Path:
        destination = Path(output_path)
        if destination.suffix.lower() != ".wav":
            raise ValueError("فایل ضبط‌شده باید با پسوند .wav ذخیره شود.")
        return destination

    @staticmethod
    def _require_sounddevice() -> None:
        if sd is None:
            raise ImportError("sounddevice نصب نیست؛ `python -m pip install -r requirements.txt` را اجرا کنید.")
