from pathlib import Path
from types import SimpleNamespace

import numpy as np

from live_stream import LiveStreamTranscriber


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, samples, **options):
        self.calls.append((samples, options))
        return iter([SimpleNamespace(start=0.0, end=1.0, text=" سلام بیمار")]), SimpleNamespace()


def test_transcribe_uses_required_farsi_settings():
    model = FakeModel()
    transcriber = LiveStreamTranscriber(model=model, sounddevice_module=object())
    result = transcriber._transcribe(np.full(16000, 4096, dtype=np.int16))

    assert result == [(0.0, 1.0, "سلام بیمار")]
    options = model.calls[0][1]
    assert options["language"] == "fa"
    assert options["task"] == "transcribe"
    assert options["temperature"] == 0.0
    assert options["beam_size"] == 1
    assert options["best_of"] == 1
    assert options["condition_on_previous_text"] is False
    assert options["repetition_penalty"] == 1.2
    assert options["vad_filter"] is True
    assert options["vad_parameters"] == {
        "min_silence_duration_ms": 600,
        "threshold": 0.2,
        "speech_pad_ms": 200,
    }
    assert options["no_speech_threshold"] == 0.1
    assert options["log_prob_threshold"] == -0.8


def test_transcribe_skips_low_amplitude_audio():
    model = FakeModel()
    transcriber = LiveStreamTranscriber(model=model, sounddevice_module=object())

    assert transcriber._transcribe(np.zeros(16000, dtype=np.int16)) == []
    assert model.calls == []


def test_sliding_window_finalizes_and_shifts_buffer(tmp_path: Path):
    model = FakeModel()
    transcriber = LiveStreamTranscriber(
        model=model,
        sounddevice_module=object(),
        max_context_seconds=2,
        finalize_seconds=1,
        output_path=tmp_path / "result.txt",
    )
    with transcriber._lock:
        transcriber._chunks = [np.full(3 * 16_000, 4096, dtype=np.int16)]

    transcriber._process_live_window(force=True)

    assert transcriber._finalized_text == ["سلام بیمار"]
    assert sum(chunk.size for chunk in transcriber._chunks) == 2 * 16_000
