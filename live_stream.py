"""رونویسی زنده فارسی با بافر افزایشی و پنجره لغزان تأخیری."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore[assignment]

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment,misc]


class LiveTranscriptionError(RuntimeError):
    """خطای قابل فهم در ضبط یا رونویسی زنده."""


class LiveStreamTranscriber:
    """ضبط پیوسته و رونویسی افزایشی با حفظ کانتکست فارسی."""

    def __init__(
        self,
        model_path: str | Path = "models/small",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "fa",
        sample_rate: int = 16_000,
        update_interval_seconds: float = 3.5,
        max_context_seconds: float = 90.0,
        finalize_seconds: float = 45.0,
        output_path: str | Path = "result.txt",
        model: Any | None = None,
        sounddevice_module: Any | None = None,
    ) -> None:
        if sample_rate <= 0 or update_interval_seconds <= 0:
            raise ValueError("sample_rate و update_interval_seconds باید بزرگ‌تر از صفر باشند.")
        if not 1 <= finalize_seconds < max_context_seconds:
            raise ValueError("finalize_seconds باید کمتر از max_context_seconds باشد.")
        if device not in {"cuda", "cpu"}:
            raise ValueError("device باید cuda یا cpu باشد.")

        self.model_path = str(model_path)
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.sample_rate = sample_rate
        self.update_interval_seconds = update_interval_seconds
        self.max_context_seconds = max_context_seconds
        self.finalize_seconds = finalize_seconds
        self.output_path = Path(output_path)
        self._model = model
        self._sd = sounddevice_module if sounddevice_module is not None else sd
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._finalized_text: list[str] = []
        self._last_live_text = ""

    def load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if WhisperModel is None:
            raise ImportError("faster-whisper نصب نیست؛ requirements.txt را نصب کنید.")
        try:
            self._model = WhisperModel(
                self.model_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            raise LiveTranscriptionError(
                f"بارگذاری مدل محلی {self.model_path} روی {self.device} شکست خورد."
            ) from exc
        return self._model

    def run(self) -> Path:
        """تا Ctrl+C ضبط می‌کند، متن زنده را نشان می‌دهد و result.txt می‌سازد."""
        self._require_sounddevice()
        self.load_model()
        stream = None
        try:
            stream = self._sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            )
            stream.start()
            print("ضبط و رونویسی زنده شروع شد؛ برای پایان Ctrl+C را بزنید.", flush=True)
            while not self._stop_event.wait(self.update_interval_seconds):
                self._process_live_window()
        except KeyboardInterrupt:
            print("\nدر حال نهایی‌سازی متن...", flush=True)
        except Exception as exc:
            raise LiveTranscriptionError("ضبط زنده شکست خورد.") from exc
        finally:
            self._stop_event.set()
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

        self._process_live_window(force=True)
        final_text = self._join_text(self._finalized_text + ([self._last_live_text] if self._last_live_text else []))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(final_text, encoding="utf-8", newline="\n")
        print(f"رونویسی نهایی در {self.output_path} ذخیره شد.", flush=True)
        return self.output_path

    def _audio_callback(self, indata: np.ndarray, _frames: int, _time: Any, _status: Any) -> None:
        with self._lock:
            self._chunks.append(np.array(indata[:, 0], dtype=np.int16, copy=True))

    def _snapshot(self) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return np.empty(0, dtype=np.int16)
            return np.concatenate(self._chunks)

    def _process_live_window(self, force: bool = False) -> None:
        samples = self._snapshot()
        if samples.size == 0:
            return
        if not force and samples.size < int(self.sample_rate * self.update_interval_seconds):
            return

        segments = self._transcribe(samples)
        context_limit = int(self.sample_rate * self.max_context_seconds)
        if samples.size > context_limit:
            cut = int(self.sample_rate * self.finalize_seconds)
            finalized = [text for start, end, text in segments if end <= self.finalize_seconds and text]
            if finalized:
                self._finalized_text.extend(finalized)
            with self._lock:
                remaining = np.concatenate(self._chunks)[cut:]
                self._chunks = [remaining] if remaining.size else []
            segments = self._transcribe(remaining) if remaining.size else []

        self._last_live_text = self._join_text([text for _start, _end, text in segments if text])
        self._render_live(self._join_text(self._finalized_text + ([self._last_live_text] if self._last_live_text else [])))

    def _transcribe(self, samples: np.ndarray) -> list[tuple[float, float, str]]:
        if samples.size == 0:
            return []
        try:
            # # تبدیل به فلوت ۳۲ بیت استاندارد
            # audio_data = samples.astype(np.float32) / 32768.0

            # # # اگر دامنه صدا خیلی ضعیف بود (سکوت مطلق یا نویز ناچیز)، اصلاً پردازش نکن
            # # if np.max(np.abs(audio_data)) < 0.02:
            # #     return []


            # segments, _info = self._model.transcribe(
            #     # samples.astype(np.float32) / 32768.0,
            #     audio_data,
            #     language=self.language,
            #     task="transcribe",
            #     temperature=0.0,
            #     beam_size=3,
            #     best_of=3,
            #     condition_on_previous_text=False,
            #     repetition_penalty=1.1,        # ضد تکرار هجاها و کلمات
            #     # no_repeat_ngram_size=3,        # جلوگیری از تکرار عبارت‌های ۳ کلمه‌ای
            #     vad_filter=True,
            #     vad_parameters=dict(min_silence_duration_ms=900, threshold=0.4, speech_pad_ms=200),
            # )

            # # تبدیل به فلوت ۳۲ بیت استاندارد
            # audio_data = samples.astype(np.float32) / 32768.0

            # # ۱. فعال کردن فیلتر سکوت/نویز مطلق میکروفون (از کامنت خارج شد)
            # if np.max(np.abs(audio_data)) < 0.02:
            #     return []

            # segments, _info = self._model.transcribe(
            #     audio_data,
            #     language=self.language,
            #     task="transcribe",
            #     temperature=0.0,
            #     beam_size=1,            # تغییر به ۱: در لایو بسیار سریع‌تر و با توهم کمتر عمل می‌کند
            #     best_of=1,              # هماهنگ با beam_size=1
            #     condition_on_previous_text=False,
            #     repetition_penalty=1.1,
            #     vad_filter=True,
            #     vad_parameters=dict(
            #         min_silence_duration_ms=500,  # کاهش به ۵۰۰ میلی‌ثانیه برای برش سریع‌تر سکوت‌ها
            #         threshold=0.5,                # استاندارد تشخیص گفتار واقعی
            #         speech_pad_ms=200
            #     ),
            #     no_speech_threshold=0.6,          # مهم: اگر سکوت بود، کلمه تولید نکن
            #     log_prob_threshold=-1.0           # مهم: اگر به متن اطمینان نداری ردش کن
            # )



            # # تبدیل به اعشاری
            # audio_data = samples.astype(np.float32) / 32768.0

            # # ۱. افزایش آستانه سکوت (اگر نویز فن یا هوا هست، عدد را روی 0.03 یا 0.04 بگذارید)
            # # با این خط، در سکوت هیچ پردازشی انجام نمی‌شود
            # if np.max(np.abs(audio_data)) < 0.05:
            #     return []

            # segments, _info = self._model.transcribe(
            #     audio_data,
            #     language=self.language,
            #     task="transcribe",
            #     temperature=0.0,
            #     beam_size=1,
            #     best_of=1,
            #     # جلوگیری از به خاطر سپردن توهم قبلی
            #     condition_on_previous_text=False,
            #     repetition_penalty=1.2,          # افزایش جریمه تکرار
            #     vad_filter=True,
            #     vad_parameters=dict(
            #         min_silence_duration_ms=600,
            #         threshold=0.65,              # افزایش حساسیت: صدا باید شفاف باشد تا گفتار تلقی شود (پیش‌فرض 0.5 بود)
            #         speech_pad_ms=150
            #     ),
            #     # این دو گزینه در زمان سکوت جلوی کلمه‌سازی الکی را می‌گیرند:
            #     no_speech_threshold=0.5,         # اگر احتمال سکوت بالای ۵۰ درصد بود متن را دور بریز
            #     log_prob_threshold=-0.8,         # اگر مدل در درستی کلمات شک دارد، نادیده‌اش بگیر
            #     # # پرامپت زمینه برای ارتقای چشمگیر دقت عبارات پزشکی و جلوگیری از کلمات چرت:
            #     # initial_prompt="گفتگوی بالینی و ویزیت پزشکی میان پزشک و بیمار، شرح حال، علائم بیماری و نام داروها."
            # )



            # تبدیل به اعشاری
            audio_data = samples.astype(np.float32) / 32768.0

            # ۱. افزایش آستانه سکوت (اگر نویز فن یا هوا هست، عدد را روی 0.03 یا 0.04 بگذارید)
            # با این خط، در سکوت هیچ پردازشی انجام نمی‌شود
            if np.max(np.abs(audio_data)) < 0.05:
                return []

            segments, _info = self._model.transcribe(
                audio_data,
                language=self.language,
                task="transcribe",
                temperature=0.0,
                beam_size=1,
                best_of=1,
                # جلوگیری از به خاطر سپردن توهم قبلی
                condition_on_previous_text=False,
                repetition_penalty=1.2,          # افزایش جریمه تکرار
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=600,
                    threshold=0.2,              # افزایش حساسیت: صدا باید شفاف باشد تا گفتار تلقی شود (پیش‌فرض 0.5 بود)
                    speech_pad_ms=200
                ),
                # این دو گزینه در زمان سکوت جلوی کلمه‌سازی الکی را می‌گیرند:
                no_speech_threshold=0.1,         # اگر احتمال سکوت بالای ۵۰ درصد بود متن را دور بریز
                log_prob_threshold=-0.8,         # اگر مدل در درستی کلمات شک دارد، نادیده‌اش بگیر
                # # پرامپت زمینه برای ارتقای چشمگیر دقت عبارات پزشکی و جلوگیری از کلمات چرت:
                # initial_prompt="گفتگوی بالینی و ویزیت پزشکی میان پزشک و بیمار، شرح حال، علائم بیماری و نام داروها."
            )



            
            return [(float(item.start), float(item.end), str(item.text).strip()) for item in segments]
        except Exception as exc:
            raise LiveTranscriptionError("رونویسی قطعه زنده شکست خورد.") from exc

    @staticmethod
    def _join_text(parts: list[str]) -> str:
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()

    @staticmethod
    def _render_live(text: str) -> None:
        # # یک خط بازنویسی می‌شود و متن پزشکی خام در لاگ ذخیره نمی‌گردد.
        # sys.stdout.write("\r\033[2K" + text.replace("\n", " "))
        # sys.stdout.flush()
        if not text.strip():
            return
        # ذخیره آنی در فایل متنی با انکودینگ استاندارد فارسی
        with open("live_output.txt", "w", encoding="utf-8") as f:
            f.write(text)
        
        # فقط یک پیام کوتاه در ترمینال بدهد که در حال ضبط است
        print(".", end="", flush=True)

    def _require_sounddevice(self) -> None:
        if self._sd is None:
            raise ImportError("sounddevice نصب نیست؛ requirements.txt را نصب کنید.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Persian transcription with delayed sliding context")
    parser.add_argument("--model-path", default="models/small")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", default="result.txt")
    parser.add_argument("--interval", type=float, default=3.5)
    args = parser.parse_args()
    try:
        LiveStreamTranscriber(
            model_path=args.model_path,
            device=args.device,
            update_interval_seconds=args.interval,
            output_path=args.output,
        ).run()
    except (ImportError, ValueError, LiveTranscriptionError) as exc:
        print(f"خطا: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
