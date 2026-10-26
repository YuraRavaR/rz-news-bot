"""Failures that are not the Article's fault.

Every such failure makes the Article Deferred: nothing is written to storage, the
run report shows it with its own icon and message, and the next Run picks it up
again without spending its Error attempts. Some of them also stop the Run early.
"""

from typing import ClassVar

from rz_flow.models import StopSignal


class DeferralError(Exception):
    """Base for every failure that Defers the Article instead of recording an Error.

    Subclasses set the run-report icon and the stop signal; the run-report message
    defaults to the exception text.
    """

    report_icon: ClassVar[str] = "🔄"
    stop_signal: ClassVar[StopSignal] = StopSignal.CONTINUE

    @property
    def report_message(self) -> str:
        return str(self)


class PublishingRecordError(DeferralError):
    """The Publishing record could not be written, so nothing was sent.

    Posting without the write-ahead record could produce a Duplicate, so the Run
    stops rather than spend Gemini quota on Articles it cannot post.
    """

    report_icon = "💾"
    stop_signal = StopSignal.PUBLISHING_BLOCKED

    @property
    def report_message(self) -> str:
        return f"storage unavailable, nothing sent ({self}) — will retry next run"
