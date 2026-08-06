"""
HelioAtlas Explorer
Geometry and mathematical utilities.

This module contains pure mathematical helper functions.
It has no dependency on matplotlib or the GUI.
"""

import numpy as np
import pandas as pd
from astropy.time import Time


def ensure_utc(obj):
   
    s = pd.to_datetime(obj, errors="coerce", utc=True)
    if isinstance(s, pd.DatetimeIndex):
        return s.tz_convert("UTC") if s.tz is not None else s.tz_localize("UTC")
    return s.dt.tz_convert("UTC") if s.dt.tz is not None else s.dt.tz_localize("UTC")


def parse_horizons_datetime_str(series):
    s = (series.astype(str)
         .str.replace("A.D. ", "", regex=False)
         .str.replace(" TDB", "", regex=False)
         .str.replace(" UT",  "", regex=False)
         .str.replace(" UTC", "", regex=False))
    try:
        return pd.to_datetime(s, format="%Y-%b-%d %H:%M:%S.%f", utc=True)
    except Exception:
        return pd.to_datetime(s, utc=True)


def horizons_times_to_utc(tbl):
    if "datetime_str" in tbl.colnames:
        try:
            return parse_horizons_datetime_str(tbl["datetime_str"].astype(str))
        except Exception:
            pass

    for cand in ("JD", "datetime_jd"):
        if cand in tbl.colnames:
            t = Time(np.array(tbl[cand], dtype=float), format="jd", scale="tdb")
            return pd.to_datetime(t.utc.datetime)

    raise RuntimeError("No time column in HORIZONS table")


def angle_deg_vectorized(u_x, u_y, u_z, v_x, v_y, v_z):
    dot = u_x * v_x + u_y * v_y + u_z * v_z
    nu = np.sqrt(u_x**2 + u_y**2 + u_z**2)
    nv = np.sqrt(v_x**2 + v_y**2 + v_z**2)
    cosang = np.clip(dot / (nu * nv), -1.0, 1.0)
    cosang[nu * nv == 0] = np.nan
    return np.degrees(np.arccos(cosang))

    

def _fill_zero_signs(s: np.ndarray) -> np.ndarray:
    s = s.astype(float)
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i-1]
    for i in range(len(s)-2, -1, -1):
        if s[i] == 0:
            s[i] = s[i+1]
    return s


def orbit_plot_limit(
    enabled,
    earth_radius,
    mars_radius,
    jupiter_radius,
    saturn_radius,
):
    """
    Compute the plotting radius from the outermost enabled planet.
    """

    radii = [earth_radius]

    if enabled.get("Mars", False):
        radii.append(mars_radius)

    if enabled.get("Jupiter", False):
        radii.append(jupiter_radius)

    if enabled.get("Saturn", False):
        radii.append(saturn_radius)

    outer = max(radii)

    return max(1.8, outer * 1.08)



def outer_enabled_orbit(
    enabled,
    mars_radius,
    jupiter_radius,
    saturn_radius,
):
    """
    Return the largest enabled planetary orbit.
    """

    radii = []

    if enabled.get("Mars", False):
        radii.append(mars_radius)

    if enabled.get("Jupiter", False):
        radii.append(jupiter_radius)

    if enabled.get("Saturn", False):
        radii.append(saturn_radius)

    if not radii:
        return jupiter_radius

    return max(radii)
