from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from audio_preprocessor import AudioPreprocessor


def _write_test_wav(path: Path) -> None:
    sample_rate = 8_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    signal = 0.04 * np.sin(2 * np.pi * 440 * t)
    wavfile.write(path, sample_rate, np.rint(signal * 32767).astype(np.int16))


def test_preprocess_creates_mono_16khz_wav(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    output = tmp_path / "output.wav"
    _write_test_wav(source)

    result = AudioPreprocessor().preprocess(source, output)
    sample_rate, data = wavfile.read(result)

    assert result == output
    assert sample_rate == 16_000
    assert data.dtype == np.int16
    assert data.ndim == 1
    assert data.size > 0


def test_preprocess_converts_stereo_to_mono(tmp_path: Path) -> None:
    source = tmp_path / "stereo.wav"
    sample_rate = 16_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    left = 0.04 * np.sin(2 * np.pi * 440 * t)
    right = 0.02 * np.sin(2 * np.pi * 660 * t)
    stereo = np.column_stack((left, right))
    wavfile.write(source, sample_rate, np.rint(stereo * 32767).astype(np.int16))

    result = AudioPreprocessor().preprocess(source)
    output_sample_rate, output_data = wavfile.read(result)

    assert result.name == "stereo_processed.wav"
    assert output_sample_rate == 16_000
    assert output_data.ndim == 1


def test_missing_input_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AudioPreprocessor().preprocess(tmp_path / "missing.wav")


def test_unsupported_extension_raises_value_error(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("not audio", encoding="utf-8")
    with pytest.raises(ValueError, match="WAV یا MP3"):
        AudioPreprocessor().preprocess(source)


def test_corrupt_audio_raises_value_error(tmp_path: Path) -> None:
    source = tmp_path / "broken.wav"
    source.write_bytes(b"not a wav file")
    with pytest.raises(ValueError, match="decode"):
        AudioPreprocessor().preprocess(source)
