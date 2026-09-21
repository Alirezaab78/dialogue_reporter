from pathlib import Path
from types import SimpleNamespace

import pytest

import transcriber
from transcriber import Transcriber, TranscriptionResult


class FakeModel:
    def transcribe(self, _path: str, **options):
        assert options["vad_filter"] is True
        return iter(
            [
                SimpleNamespace(start=0.0, end=1.25, text=" سلام "),
                SimpleNamespace(start=1.5, end=2.75, text="حال شما؟"),
            ]
        ), SimpleNamespace(duration=2.75)


def test_defaults_select_valid_device_and_compute_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Transcriber, "_cuda_available", staticmethod(lambda: False))
    instance = Transcriber()
    assert instance.device == "cpu"
    assert instance.compute_type == "int8"


def test_cuda_defaults_to_float16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Transcriber, "_cuda_available", staticmethod(lambda: True))
    instance = Transcriber(device="auto")
    assert instance.device == "cuda"
    assert instance.compute_type == "float16"


def test_transcribe_returns_structured_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"placeholder")
    monkeypatch.setattr(transcriber, "WhisperModel", lambda *args, **kwargs: FakeModel())
    instance = Transcriber(device="cpu")

    result = instance.transcribe(audio)

    assert isinstance(result, TranscriptionResult)
    assert result.hardware == "cpu"
    assert len(result.segments) == 2
    assert result.segments[0].start == 0.0
    assert result.full_text == "سلام حال شما؟"
    assert result.processing_time_seconds >= 0
    assert result.to_dict()["segments"][1]["end"] == 2.75


def test_missing_audio_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        Transcriber(device="cpu").transcribe("does-not-exist.wav")


def test_invalid_device_raises_value_error() -> None:
    with pytest.raises(ValueError):
        Transcriber(device="tpu")  # type: ignore[arg-type]
