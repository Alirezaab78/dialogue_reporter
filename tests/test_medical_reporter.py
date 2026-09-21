import logging
from types import SimpleNamespace

import pytest

from medical_reporter import MedicalReporter


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeClient:
    def __init__(self, response):
        self.completions = FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def test_defaults_and_custom_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reporter = MedicalReporter(client=object())
    assert reporter.model == "gpt-5.6-terra"
    assert reporter.base_url == "https://api.openai.com/v1"
    assert reporter.timeout == 30.0
    assert reporter.max_retries == 2


def test_generate_report_returns_text_and_uses_safe_metadata_logging(caplog: pytest.LogCaptureFixture) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="گزارش رسمی پزشکی"))],
        usage=SimpleNamespace(prompt_tokens=21, completion_tokens=8),
    )
    client = FakeClient(response)
    reporter = MedicalReporter(client=client, model="test-model")
    transcript = "اطلاعات محرمانه بیمار 123"

    with caplog.at_level(logging.INFO):
        result = reporter.generate_report(transcript)

    assert result == "گزارش رسمی پزشکی"
    assert client.completions.kwargs["model"] == "test-model"
    assert client.completions.kwargs["messages"][1]["content"] == transcript
    assert "prompt_tokens=21" in caplog.text
    assert "completion_tokens=8" in caplog.text
    assert transcript not in caplog.text
    assert result not in caplog.text


def test_empty_transcript_is_rejected() -> None:
    with pytest.raises(ValueError):
        MedicalReporter(client=object()).generate_report("  ")


def test_api_failure_does_not_log_transcript(caplog: pytest.LogCaptureFixture) -> None:
    class FailingCompletions:
        def create(self, **_kwargs):
            raise RuntimeError("upstream failed")

    reporter = MedicalReporter(client=SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions())))
    transcript = "محرمانه-بیمار"
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        reporter.generate_report(transcript)
    assert transcript not in caplog.text
