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
