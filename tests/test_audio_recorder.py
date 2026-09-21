from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

import audio_recorder
from audio_recorder import AudioRecorder, AudioRecordingError


class FakeSoundDevice:
    class Default:
        device = (3, 4)

    default = Default()

    def __init__(self):
        self.wait_called = False

    def query_devices(self, kind=None):
        assert kind == "input"
        return {"name": "Mock Microphone", "max_input_channels": 1}

    def rec(self, frames, samplerate, channels, dtype, device):
        assert frames == 16_000
        assert samplerate == 16_000
        assert channels == 1
        assert dtype == "int16"
        assert device == 3
        return np.zeros((frames, channels), dtype=np.int16)

    def wait(self):
        self.wait_called = True


def test_record_writes_standard_mono_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_sd = FakeSoundDevice()
    monkeypatch.setattr(audio_recorder, "sd", fake_sd)
    output = tmp_path / "recorded.wav"

    result = AudioRecorder().record(output, duration_seconds=1.0)
    sample_rate, data = wavfile.read(result)

    assert fake_sd.wait_called is True
    assert sample_rate == 16_000
    assert data.dtype == np.int16
    assert data.ndim == 1
    assert data.size == 16_000


def test_invalid_output_and_duration_are_rejected(tmp_path: Path) -> None:
    recorder = AudioRecorder()
    with pytest.raises(ValueError):
        recorder.record(tmp_path / "recorded.mp3", duration_seconds=1)
    with pytest.raises(ValueError):
        recorder.record(tmp_path / "recorded.wav", duration_seconds=0)


def test_missing_sounddevice_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_recorder, "sd", None)
    with pytest.raises(ImportError):
        AudioRecorder().record(tmp_path / "recorded.wav", duration_seconds=1)


def test_recording_error_is_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingSoundDevice(FakeSoundDevice):
        def rec(self, **_kwargs):
            raise OSError("microphone unavailable")

    monkeypatch.setattr(audio_recorder, "sd", FailingSoundDevice())
    with pytest.raises(AudioRecordingError, match="ضبط زمان‌دار"):
        AudioRecorder().record(tmp_path / "recorded.wav", duration_seconds=1)
