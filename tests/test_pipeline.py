from pathlib import Path

from main import run_pipeline


class FakeTranscriber:
    def transcribe(self, _audio_path):
        return type("Result", (), {"full_text": "بیمار از سردرد شکایت دارد."})()


class FakeReporter:
    def generate_report(self, transcript: str) -> str:
        assert transcript == "بیمار از سردرد شکایت دارد."
        return "گزارش پزشکی\nشکایت اصلی: سردرد"


def test_pipeline_writes_utf8_transcript_and_report(tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fake audio")
    transcript_path = tmp_path / "result.txt"
    report_path = tmp_path / "final_medical_report.txt"

    output, elapsed = run_pipeline(
        audio,
        report_output_path=report_path,
        transcript_output_path=transcript_path,
        transcriber=FakeTranscriber(),
        reporter=FakeReporter(),
    )

    assert output == report_path
    assert elapsed >= 0
    assert transcript_path.read_text(encoding="utf-8") == "بیمار از سردرد شکایت دارد."
    assert report_path.read_text(encoding="utf-8") == "گزارش پزشکی\nشکایت اصلی: سردرد"
