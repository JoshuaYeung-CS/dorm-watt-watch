"""Sun elevation, used to build the "it is dark indoors" signal that identifies lighting.

Standard low-precision solar position (NOAA spreadsheet equations); accurate to well under a
degree, which is plenty for deciding whether a room needs lights.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

ITHACA = (42.444, -76.502)


def utc_offset_hours(hours: np.ndarray, tz: str = "America/New_York") -> np.ndarray:
    z = ZoneInfo(tz)
    # Offsets change only at DST boundaries; compute per day to keep it cheap.
    days = hours.astype("datetime64[D]")
    uniq, inv = np.unique(days, return_inverse=True)
    offs = []
    for d in uniq:
        dt = datetime.fromisoformat(str(d)).replace(hour=12, tzinfo=z)
        offs.append(dt.utcoffset().total_seconds() / 3600.0)
    return np.array(offs)[inv]


def sun_elevation_deg(hours: np.ndarray, lat: float = ITHACA[0], lon: float = ITHACA[1],
                      tz: str = "America/New_York", minute: float = 30.0) -> np.ndarray:
    """Elevation of the sun at the middle of each local clock hour."""
    utc = hours.astype("datetime64[m]") + np.timedelta64(int(minute), "m") \
        - (utc_offset_hours(hours, tz) * 60).astype("timedelta64[m]")
    jd = utc.astype("datetime64[s]").astype(float) / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    l0 = np.deg2rad((280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360)
    m = np.deg2rad(357.52911 + jc * (35999.05029 - 0.0001537 * jc))
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    c = np.deg2rad(np.sin(m) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
                   + np.sin(2 * m) * (0.019993 - 0.000101 * jc) + np.sin(3 * m) * 0.000289)
    true_long = l0 + c
    omega = np.deg2rad(125.04 - 1934.136 * jc)
    app_long = true_long - np.deg2rad(0.00569 + 0.00478 * np.sin(omega))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = np.deg2rad(eps0 + 0.00256 * np.cos(omega))
    decl = np.arcsin(np.sin(eps) * np.sin(app_long))
    y = np.tan(eps / 2) ** 2
    eqt = 4 * np.rad2deg(y * np.sin(2 * l0) - 2 * e * np.sin(m) + 4 * e * y * np.sin(m) * np.cos(2 * l0)
                         - 0.5 * y * y * np.sin(4 * l0) - 1.25 * e * e * np.sin(2 * m))  # minutes
    utc_min = (utc - utc.astype("datetime64[D]")).astype(float)
    tst = (utc_min + eqt + 4 * lon) % 1440
    ha = np.deg2rad(np.where(tst / 4 < 0, tst / 4 + 180, tst / 4 - 180))
    phi = np.deg2rad(lat)
    cos_z = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    return 90.0 - np.rad2deg(np.arccos(np.clip(cos_z, -1, 1)))


def darkness(elev_deg: np.ndarray, full_dark: float = -2.0, full_light: float = 12.0) -> np.ndarray:
    """0 when the sun is high enough to light a room, 1 after sunset, linear in between."""
    return np.clip((full_light - elev_deg) / (full_light - full_dark), 0.0, 1.0)
