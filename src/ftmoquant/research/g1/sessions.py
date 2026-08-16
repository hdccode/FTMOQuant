"""Central timezone-safe G1 session eligibility definitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo


class SessionDefinitionError(ValueError):
    """Raised for naive timestamps or unknown session semantics."""


class SessionId(StrEnum):
    ALL = "all"
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    LONDON_NEW_YORK_OVERLAP = "london_new_york_overlap"


@dataclass(frozen=True, slots=True)
class LocalSession:
    session_id: SessionId
    timezone: str
    local_start: time
    local_end: time

    def contains(self, timestamp: datetime) -> bool:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise SessionDefinitionError("session timestamps must be timezone-aware")
        local_time = (
            timestamp.astimezone(ZoneInfo(self.timezone)).timetz().replace(tzinfo=None)
        )
        if self.local_start < self.local_end:
            return self.local_start <= local_time < self.local_end
        return local_time >= self.local_start or local_time < self.local_end


ASIAN_SESSION = LocalSession(SessionId.ASIAN, "Asia/Tokyo", time(9), time(18))
LONDON_SESSION = LocalSession(SessionId.LONDON, "Europe/London", time(8), time(17))
NEW_YORK_SESSION = LocalSession(
    SessionId.NEW_YORK, "America/New_York", time(8), time(17)
)


def is_session_eligible(timestamp: datetime, session_id: SessionId) -> bool:
    """Evaluate sessions in local civil time, including both DST calendars."""

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise SessionDefinitionError("session timestamps must be timezone-aware")
    if session_id is SessionId.ALL:
        return True
    if session_id is SessionId.ASIAN:
        return ASIAN_SESSION.contains(timestamp)
    if session_id is SessionId.LONDON:
        return LONDON_SESSION.contains(timestamp)
    if session_id is SessionId.NEW_YORK:
        return NEW_YORK_SESSION.contains(timestamp)
    if session_id is SessionId.LONDON_NEW_YORK_OVERLAP:
        return LONDON_SESSION.contains(timestamp) and NEW_YORK_SESSION.contains(
            timestamp
        )
    raise SessionDefinitionError(f"unknown session: {session_id}")
