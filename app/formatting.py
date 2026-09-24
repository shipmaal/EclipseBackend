"""Presentation of the core's times: hours from T0 as clock and offset strings."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def dec_to_hms(t_hours: float, t0_hours: float = 0.0) -> tuple[int, int, float]:
    """Decimal hours (offset by ``t0_hours``) -> (hour, minute, second)."""
    decimal_hour = t0_hours + t_hours
    hour = int(decimal_hour)
    minute_f = (decimal_hour - hour) * 60.0
    minute = int(minute_f)
    second = (minute_f - minute) * 60.0
    if round(second) == 60:
        minute += 1
        second = 0.0
    if minute == 60:
        hour += 1
        minute = 0
    return hour, minute, second


def format_offset(t_hours: float) -> str:
    """Signed offset from T0 as ``+HH:MM:SS.s`` / ``-HH:MM:SS.s`` (item R4)."""
    sign = "-" if t_hours < 0 else "+"
    h, m, s = dec_to_hms(abs(t_hours))
    return f"{sign}{h:02d}:{m:02d}:{s:04.1f}"


def format_clock(t0_utc: str, t_hours: float) -> str:
    """Absolute UTC wall-clock ``HH:MM:SS`` at ``t_hours`` from ISO ``t0_utc`` (item R4),
    rounded to the nearest second (``strftime`` alone truncates: -0.5 s on
    average, item C7).

    ``datetime`` handles date rollover across midnight.
    """
    t = datetime.fromisoformat(t0_utc).replace(tzinfo=timezone.utc) + timedelta(hours=t_hours)
    return (t + timedelta(microseconds=500_000)).replace(microsecond=0).strftime("%H:%M:%S")
