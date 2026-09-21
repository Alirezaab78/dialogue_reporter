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
    result = transcriber._transcribe(np.zeros(16000, dtype=np.int16))

    assert result == [(0.0, 1.0, "سلام بیمار")]
    options = model.calls[0][1]
    assert options == {"language": "fa", "beam_size": 3, "vad_filter": True}


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
        transcriber._chunks = [np.zeros(3 * 16_000, dtype=np.int16)]

    transcriber._process_live_window(force=True)

    assert transcriber._finalized_text == ["سلام بیمار"]
    assert sum(chunk.size for chunk in transcriber._chunks) == 2 * 16_000
