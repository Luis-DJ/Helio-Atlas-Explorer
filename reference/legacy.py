#!/usr/bin/env python3
"""
HelioAtlas — Jupiter/Mars/Saturn opposition/conjunction visualiser (interactive)
Classic Ecliptic View + Dynamic View Limits + Per-planet visibility control

Fixes included:
- Restores full Fig 2 + slider UI (ax2/ax2b/etc. are defined).
- Fixes BUG: Mars event markers were incorrectly registered to Jupiter.
- Legend + readout box only show enabled planets.
- Dynamic view limits: plot framing uses the outermost enabled planet (Earth always included).
"""

from __future__ import annotations

import sys
import matplotlib

# --- Backend selection (EXE-friendly logic) ---
if getattr(sys, "frozen", False):
    matplotlib.use("Qt5Agg")
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

try:
    from astroquery.jplhorizons import Horizons
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("astroquery is required. Install with: pip install astroquery") from e

from astropy.time import Time

APP_NAME = "HelioAtlas"
APP_VERSION = "0.9.2.18-jupiter-opp-conj-classic+mars+saturn-dynamicview-fixed"

# -------------------- CONFIGURATION --------------------
# Planet toggles (set these before running)
SHOW_PLANET_JUPITER = True
SHOW_PLANET_MARS    = True
SHOW_PLANET_SATURN  = True

# Timeline
START = "2025-07-01"
STOP  = "2037-03-31"
STEP  = "1d"
MARK  = "2026-01-10"  # a good default anchor near Jupiter opposition
DATE_FMT = "%d-%b-%Y"

# Reference orbit radii (AU)
EARTH_ORBIT_AU   = 1.0
MARS_ORBIT_AU    = 1.524
JUPITER_ORBIT_AU = 5.204
SATURN_ORBIT_AU  = 9.582

# Visual sizes
SUN_GLOW_SIZE   = 220
SUN_CORE_SIZE   = 70
EARTH_SIZE      = 60
MARS_SIZE       = 65
JUPITER_SIZE    = 80
SATURN_SIZE     = 85

EVENT_SIZE      = 95
EVENT_OPP_COLOR  = "white"
EVENT_CONJ_COLOR = "crimson"
EVENT_MARKER_LW  = 1.8
SHOW_EVENT_LABELS = True

# Solar-glare (elongation) bands / mask
SHOW_SOLAR_GLARE_MASK = True
ELONG_HIDE_DEG        = 10.0
ELONG_DIFFICULT_DEG   = 15.0
ELONG_USABLE_DEG      = 25.0
ELONG_EXCELLENT_DEG   = 60.0

GLARE_SHADE_BELOW_DEG = ELONG_DIFFICULT_DEG
GLARE_SHADE_COLOR     = "crimson"
GLARE_SHADE_ALPHA     = 0.10

# Connector styling by elongation
CONNECTOR_LW_DIFFICULT = 3.0   # 10–15°
CONNECTOR_LW_MARGINAL  = 3.2   # 15–25°
CONNECTOR_LW_USABLE    = 3.2   # 25–60°
CONNECTOR_LW_EXCELLENT = 2.8   # >=60°
HIDE_CONNECTOR_BELOW_DEG = ELONG_HIDE_DEG

# Colors
SUN_CORE_COLOR      = "orangered"
SUN_GLOW_COLOR      = "yellow"
EARTH_COLOR         = "dodgerblue"
MARS_COLOR          = "tomato"
JUPITER_COLOR       = "orange"
SATURN_COLOR        = "gold"
GUIDE_COLOR         = "gray"
BACKGROUND_COLOR    = "#0d1117"


# -------------------- Helpers --------------------
def maximize_figure(fig):
    """Best-effort maximize (Windows; works for many Tk/Qt backends)."""
    try:
        manager = fig.canvas.manager
        win = getattr(manager, "window", None)
        if win is not None and hasattr(win, "showMaximized"):  # Qt
            win.showMaximized()
            return
        if win is not None and hasattr(win, "state"):          # Tk
            win.state("zoomed")
            return
    except Exception:
        pass


def ensure_utc(obj):
    """Ensure pandas datetime objects are timezone-aware UTC."""
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


def find_conj_opp_events(elong_deg: np.ndarray, conj_thresh_deg: float = 10.0, opp_thresh_deg: float = 170.0):
    """Find conjunction (local minima near 0°) and opposition (local maxima near 180°) indices."""
    e = np.asarray(elong_deg, dtype=float)
    d = np.diff(e)
    s = np.sign(d)
    s = _fill_zero_signs(s)

    mins = np.where((s[:-1] < 0) & (s[1:] > 0))[0] + 1
    maxs = np.where((s[:-1] > 0) & (s[1:] < 0))[0] + 1

    conj = [int(i) for i in mins if np.isfinite(e[i]) and e[i] <= conj_thresh_deg]
    opp  = [int(i) for i in maxs if np.isfinite(e[i]) and e[i] >= opp_thresh_deg]

    if len(conj) == 0 and np.isfinite(e).any():
        conj = [int(np.nanargmin(np.abs(e - 0.0)))]
    if len(opp) == 0 and np.isfinite(e).any():
        opp = [int(np.nanargmin(np.abs(e - 180.0)))]

    return sorted(set(conj)), sorted(set(opp))


def compute_view_lim(planet_enabled_map=None):
    """Dynamic plot limits based on the outermost enabled planet.
    Earth is always included so the frame never collapses.
    """
    if planet_enabled_map is None:
        planet_enabled_map = {}

    radii = [EARTH_ORBIT_AU]

    def _on(name: str, default: bool) -> bool:
        if isinstance(planet_enabled_map, dict):
            return bool(planet_enabled_map.get(name, default))
        return bool(default)

    if _on("Mars", SHOW_PLANET_MARS):
        radii.append(MARS_ORBIT_AU)
    if _on("Jupiter", SHOW_PLANET_JUPITER):
        radii.append(JUPITER_ORBIT_AU)
    if _on("Saturn", SHOW_PLANET_SATURN):
        radii.append(SATURN_ORBIT_AU)

    outer = max(radii) if radii else JUPITER_ORBIT_AU
    return max(1.8, outer * 1.08)


def _mask_to_spans(mask: np.ndarray):
    spans = []
    start = None
    for k, m in enumerate(mask):
        if m and start is None:
            start = k
        if (not m) and start is not None:
            spans.append((start, k-1))
            start = None
    if start is not None:
        spans.append((start, len(mask)-1))
    return spans


# -------------------- HORIZONS fetch --------------------
def fetch_heliocentric_xyz_majorbody(naif_id: str) -> pd.DataFrame:
    obj = Horizons(id=str(naif_id), id_type="majorbody", location="@sun",
                   epochs={"start": START, "stop": STOP, "step": STEP})
    tbl = obj.vectors(refplane="ecliptic")
    df = tbl.to_pandas()
    df["t"] = ensure_utc(horizons_times_to_utc(tbl))
    return df[["t", "x", "y", "z"]].sort_values("t").reset_index(drop=True)


def build_dataset() -> pd.DataFrame:
    base = pd.DataFrame({
        "t": pd.date_range(
            pd.Timestamp(START).tz_localize("UTC"),
            pd.Timestamp(STOP).tz_localize("UTC"),
            freq="1D"
        )
    }).sort_values("t").reset_index(drop=True)

    earth_hc   = fetch_heliocentric_xyz_majorbody("399")
    jupiter_hc = fetch_heliocentric_xyz_majorbody("599")
    mars_hc    = fetch_heliocentric_xyz_majorbody("499")
    saturn_hc  = fetch_heliocentric_xyz_majorbody("699")

    def asof_align(df_base: pd.DataFrame, df_src: pd.DataFrame, suffix: str) -> pd.DataFrame:
        left = df_base[["t"]].copy()
        right = df_src[["t", "x", "y", "z"]].copy().sort_values("t")
        out = pd.merge_asof(left.sort_values("t"), right, on="t",
                            direction="nearest", tolerance=pd.Timedelta("12h"))
        out = out.add_suffix(suffix)
        out.rename(columns={f"t{suffix}": "t"}, inplace=True)
        return out

    aE = asof_align(base, earth_hc, "_E")
    aJ = asof_align(base, jupiter_hc, "_J")
    aM = asof_align(base, mars_hc, "_M")
    aS = asof_align(base, saturn_hc, "_S")

    df = aE.merge(aJ, on="t").merge(aM, on="t").merge(aS, on="t").dropna().reset_index(drop=True)

    # Distances (Earth->planet)
    df["earth_jupiter_range_AU"] = np.sqrt((df["x_J"] - df["x_E"])**2 + (df["y_J"] - df["y_E"])**2 + (df["z_J"] - df["z_E"])**2)
    df["earth_mars_range_AU"]    = np.sqrt((df["x_M"] - df["x_E"])**2 + (df["y_M"] - df["y_E"])**2 + (df["z_M"] - df["z_E"])**2)
    df["earth_saturn_range_AU"]  = np.sqrt((df["x_S"] - df["x_E"])**2 + (df["y_S"] - df["y_E"])**2 + (df["z_S"] - df["z_E"])**2)

    # Earth-based elongations: angle between (Earth->Sun) and (Earth->Planet)
    df["jup_elong_deg"] = angle_deg_vectorized(-df["x_E"], -df["y_E"], -df["z_E"], df["x_J"] - df["x_E"], df["y_J"] - df["y_E"], df["z_J"] - df["z_E"])
    df["mars_elong_deg"] = angle_deg_vectorized(-df["x_E"], -df["y_E"], -df["z_E"], df["x_M"] - df["x_E"], df["y_M"] - df["y_E"], df["z_M"] - df["z_E"])
    df["sat_elong_deg"] = angle_deg_vectorized(-df["x_E"], -df["y_E"], -df["z_E"], df["x_S"] - df["x_E"], df["y_S"] - df["y_E"], df["z_S"] - df["z_E"])

    # Sun-centered separation angles for reference: angle(r_E, r_P)
    df["helio_sep_J_deg"] = angle_deg_vectorized(df["x_E"], df["y_E"], df["z_E"], df["x_J"], df["y_J"], df["z_J"])
    df["helio_sep_M_deg"] = angle_deg_vectorized(df["x_E"], df["y_E"], df["z_E"], df["x_M"], df["y_M"], df["z_M"])
    df["helio_sep_S_deg"] = angle_deg_vectorized(df["x_E"], df["y_E"], df["z_E"], df["x_S"], df["y_S"], df["z_S"])

    return df


# -------------------- Plot + slider --------------------
def main():
    df = build_dataset()
    df["t"] = ensure_utc(df["t"])
    MAX_IDX = len(df) - 1

    # Default mark
    mark_dt = pd.Timestamp(MARK, tz="UTC")
    diffs = (df["t"] - mark_dt).dt.total_seconds().abs()
    i0 = int(diffs.idxmin()) if diffs.notna().any() else len(df) // 2

    dates = df["t"].dt.tz_convert("UTC")
    days = dates.dt.tz_localize(None)  # Matplotlib-friendly naive datetime

    # Events
    j_conj, j_opp = find_conj_opp_events(df["jup_elong_deg"].to_numpy())
    m_conj, m_opp = find_conj_opp_events(df["mars_elong_deg"].to_numpy())
    s_conj, s_opp = find_conj_opp_events(df["sat_elong_deg"].to_numpy())

    # ---------- FIG 1 ----------
    fig1, ax1 = plt.subplots(figsize=(12.5, 9.5), num=f"{APP_NAME} {APP_VERSION} — Top-down View")
    fig1.patch.set_facecolor(BACKGROUND_COLOR)
    ax1.set_facecolor(BACKGROUND_COLOR)
    maximize_figure(fig1)

    # Per-planet artist registry
    planet_artists = {"Jupiter": [], "Mars": [], "Saturn": []}
    planet_enabled = {
        "Jupiter": bool(SHOW_PLANET_JUPITER),
        "Mars": bool(SHOW_PLANET_MARS),
        "Saturn": bool(SHOW_PLANET_SATURN),
    }

    def _reg(planet: str, *artists):
        if planet not in planet_artists:
            planet_artists[planet] = []
        for a in artists:
            if a is None:
                continue
            if isinstance(a, (list, tuple)):
                for aa in a:
                    if aa is not None:
                        planet_artists[planet].append(aa)
            else:
                planet_artists[planet].append(a)

    def refresh_legend():
        leg = ax1.get_legend()
        if leg is not None:
            try:
                leg.remove()
            except Exception:
                pass

        handles, labels = ax1.get_legend_handles_labels()
        items, seen = [], set()
        for h, lab in zip(handles, labels):
            if not lab or lab == "_nolegend_":
                continue
            try:
                vis = h.get_visible()
            except Exception:
                vis = True
            if not vis:
                continue
            if lab in seen:
                continue
            seen.add(lab)
            items.append((h, lab))

        if items:
            hs, ls = zip(*items)
            ax1.legend(hs, ls, loc="lower right", frameon=True, fontsize=8)

    def apply_planet_visibility():
        for planet, artists in planet_artists.items():
            enabled = planet_enabled.get(planet, True)
            for a in artists:
                try:
                    # If it's a Text label: only show if planet is enabled AND labels enabled
                    if hasattr(a, "get_text") and not hasattr(a, "get_offsets"):
                        a.set_visible(bool(enabled) and bool(SHOW_EVENT_LABELS))
                    else:
                        a.set_visible(bool(enabled))
                except Exception:
                    pass
        refresh_legend()

    def set_view_limits():
        lim = compute_view_lim(planet_enabled)
        ax1.set_xlim(-lim, lim)
        ax1.set_ylim(-lim, lim)

    def style_connector(line, elong_deg: float):
        if line is None:
            return
        if not np.isfinite(elong_deg):
            line.set_alpha(0.0)
            return
        if elong_deg < float(HIDE_CONNECTOR_BELOW_DEG):
            line.set_alpha(0.0)
            line.set_linewidth(0.0)
            return

        if elong_deg < float(ELONG_DIFFICULT_DEG):
            line.set_alpha(0.25); line.set_color("crimson"); line.set_linestyle((0, (1, 3))); line.set_linewidth(CONNECTOR_LW_DIFFICULT)
        elif elong_deg < float(ELONG_USABLE_DEG):
            line.set_alpha(0.35); line.set_color("orange");  line.set_linestyle((0, (1, 3))); line.set_linewidth(CONNECTOR_LW_MARGINAL)
        elif elong_deg < float(ELONG_EXCELLENT_DEG):
            line.set_alpha(0.55); line.set_color("yellowgreen"); line.set_linestyle((0, (1, 3))); line.set_linewidth(CONNECTOR_LW_USABLE)
        else:
            line.set_alpha(0.75); line.set_color("lime"); line.set_linestyle("solid"); line.set_linewidth(CONNECTOR_LW_EXCELLENT)

    theta = np.linspace(0, 2*np.pi, 1200)

    # Guide orbits (only for enabled planets)
    ax1.plot(np.cos(theta), np.sin(theta), "--", color=GUIDE_COLOR, alpha=0.45, lw=1.0, label="Earth orbit (1 AU)")
    if planet_enabled["Mars"]:
        h = ax1.plot(MARS_ORBIT_AU*np.cos(theta), MARS_ORBIT_AU*np.sin(theta), "--", color=GUIDE_COLOR, alpha=0.35, lw=1.0,
                     label=f"Mars orbit ({MARS_ORBIT_AU:.3f} AU)")[0]
        _reg("Mars", h)
    if planet_enabled["Jupiter"]:
        h = ax1.plot(JUPITER_ORBIT_AU*np.cos(theta), JUPITER_ORBIT_AU*np.sin(theta), "--", color=GUIDE_COLOR, alpha=0.28, lw=1.0,
                     label=f"Jupiter orbit ({JUPITER_ORBIT_AU:.3f} AU)")[0]
        _reg("Jupiter", h)
    if planet_enabled["Saturn"]:
        h = ax1.plot(SATURN_ORBIT_AU*np.cos(theta), SATURN_ORBIT_AU*np.sin(theta), "--", color=GUIDE_COLOR, alpha=0.20, lw=1.0,
                     label=f"Saturn orbit ({SATURN_ORBIT_AU:.3f} AU)")[0]
        _reg("Saturn", h)

    # Sun
    ax1.scatter([0], [0], s=SUN_GLOW_SIZE, color=SUN_GLOW_COLOR, alpha=0.22, zorder=3)
    ax1.scatter([0], [0], s=SUN_CORE_SIZE, color=SUN_CORE_COLOR, edgecolor="white", label="Sun", zorder=4)

    # Tracks
    ax1.plot(df["x_E"], df["y_E"], color=EARTH_COLOR, lw=1.5, alpha=0.55, label="Earth (HORIZONS)")
    mars_track = ax1.plot(df["x_M"], df["y_M"], color=MARS_COLOR, lw=1.8, alpha=0.50, label="Mars (HORIZONS)")[0]
    jup_track  = ax1.plot(df["x_J"], df["y_J"], color=JUPITER_COLOR, lw=2.0, alpha=0.55, label="Jupiter (HORIZONS)")[0]
    sat_track  = ax1.plot(df["x_S"], df["y_S"], color=SATURN_COLOR, lw=2.0, alpha=0.45, label="Saturn (HORIZONS)")[0]
    _reg("Mars", mars_track)
    _reg("Jupiter", jup_track)
    _reg("Saturn", sat_track)

    # Current markers
    E = ax1.scatter([df["x_E"].iloc[i0]], [df["y_E"].iloc[i0]], s=EARTH_SIZE, color=EARTH_COLOR, edgecolor="white",
                    label="_nolegend_", zorder=6)
    M = ax1.scatter([df["x_M"].iloc[i0]], [df["y_M"].iloc[i0]], s=MARS_SIZE, color=MARS_COLOR, edgecolor="white",
                    label="_nolegend_", zorder=6)
    J = ax1.scatter([df["x_J"].iloc[i0]], [df["y_J"].iloc[i0]], s=JUPITER_SIZE, color=JUPITER_COLOR, edgecolor="white",
                    label="_nolegend_", zorder=6)
    S = ax1.scatter([df["x_S"].iloc[i0]], [df["y_S"].iloc[i0]], s=SATURN_SIZE, color=SATURN_COLOR, edgecolor="white",
                    label="_nolegend_", zorder=6)
    _reg("Mars", M)
    _reg("Jupiter", J)
    _reg("Saturn", S)

    # Connectors
    lineEM = ax1.plot([df["x_E"].iloc[i0], df["x_M"].iloc[i0]], [df["y_E"].iloc[i0], df["y_M"].iloc[i0]],
                      color="white", alpha=0.55, lw=CONNECTOR_LW_USABLE, zorder=5)[0]
    lineEJ = ax1.plot([df["x_E"].iloc[i0], df["x_J"].iloc[i0]], [df["y_E"].iloc[i0], df["y_J"].iloc[i0]],
                      color="white", alpha=0.65, lw=CONNECTOR_LW_USABLE, zorder=5)[0]
    lineES = ax1.plot([df["x_E"].iloc[i0], df["x_S"].iloc[i0]], [df["y_E"].iloc[i0], df["y_S"].iloc[i0]],
                      color="white", alpha=0.45, lw=CONNECTOR_LW_USABLE, zorder=5)[0]
    _reg("Mars", lineEM)
    _reg("Jupiter", lineEJ)
    _reg("Saturn", lineES)

    if SHOW_SOLAR_GLARE_MASK:
        style_connector(lineEJ, float(df["jup_elong_deg"].iloc[i0]))
        style_connector(lineEM, float(df["mars_elong_deg"].iloc[i0]))
        style_connector(lineES, float(df["sat_elong_deg"].iloc[i0]))

    # Events (rings + optional labels), all registered to the correct planet
    def draw_events(planet: str, xcol: str, ycol: str, opp_idxs, conj_idxs, size=EVENT_SIZE, fs=8):
        # Opposition
        for k, ii in enumerate(opp_idxs):
            h = ax1.scatter(df[xcol].iloc[ii], df[ycol].iloc[ii],
                            s=size, facecolor="none", edgecolor=EVENT_OPP_COLOR, lw=EVENT_MARKER_LW, zorder=9,
                            label="Opposition" if (planet == "Jupiter" and k == 0) else "_nolegend_")
            _reg(planet, h)
            if SHOW_EVENT_LABELS:
                yoff = 0.18 if (k % 2 == 0) else 0.30
                t = ax1.text(df[xcol].iloc[ii], df[ycol].iloc[ii] + yoff,
                             f"Opp\n{df['t'].iloc[ii].strftime(DATE_FMT)}",
                             ha="center", va="bottom", fontsize=fs, color="white",
                             bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, lw=0))
                _reg(planet, t)

        # Conjunction
        for k, ii in enumerate(conj_idxs):
            h = ax1.scatter(df[xcol].iloc[ii], df[ycol].iloc[ii],
                            s=size, facecolor="none", edgecolor=EVENT_CONJ_COLOR, lw=EVENT_MARKER_LW, zorder=9,
                            label="Conjunction" if (planet == "Jupiter" and k == 0) else "_nolegend_")
            _reg(planet, h)
            if SHOW_EVENT_LABELS:
                yoff = -0.22 if (k % 2 == 0) else -0.36
                t = ax1.text(df[xcol].iloc[ii], df[ycol].iloc[ii] + yoff,
                             f"Conj\n{df['t'].iloc[ii].strftime(DATE_FMT)}",
                             ha="center", va="top", fontsize=fs, color="white",
                             bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, lw=0))
                _reg(planet, t)

    # Jupiter full-size; Mars slightly smaller; Saturn full-size
    draw_events("Jupiter", "x_J", "y_J", j_opp, j_conj, size=EVENT_SIZE, fs=8)
    draw_events("Mars",    "x_M", "y_M", m_opp, m_conj, size=EVENT_SIZE*0.85, fs=7.5)
    draw_events("Saturn",  "x_S", "y_S", s_opp, s_conj, size=EVENT_SIZE, fs=8)

    ax1.set_aspect("equal")
    set_view_limits()

    ax1.set_xlabel("x (AU)", color="white")
    ax1.set_ylabel("y (AU)", color="white")
    ax1.tick_params(axis="x", colors="white")
    ax1.tick_params(axis="y", colors="white")
    ax1.set_title(f"Earth–Planet Geometry ({APP_VERSION}) — Heliocentric ecliptic", color="white")
    ax1.grid(True, ls=":", alpha=0.35, color="white")

    # Readout (Fig 1)
    def make_readout_text(i):
        d = dates.iloc[i].strftime("%d-%b-%Y")
        parts = [f"Date: {d}"]

        if planet_enabled["Jupiter"]:
            parts.append(f"Earth–Jupiter: {df['earth_jupiter_range_AU'].iloc[i]:6.3f} AU   Elong: {df['jup_elong_deg'].iloc[i]:5.1f}°")
        if planet_enabled["Mars"]:
            parts.append(f"Earth–Mars:   {df['earth_mars_range_AU'].iloc[i]:6.3f} AU   Elong: {df['mars_elong_deg'].iloc[i]:5.1f}°")
        if planet_enabled["Saturn"]:
            parts.append(f"Earth–Saturn: {df['earth_saturn_range_AU'].iloc[i]:6.3f} AU   Elong: {df['sat_elong_deg'].iloc[i]:5.1f}°")

        helio_bits = []
        if planet_enabled["Jupiter"]:
            helio_bits.append(f"J: {df['helio_sep_J_deg'].iloc[i]:4.1f}°")
        if planet_enabled["Mars"]:
            helio_bits.append(f"M: {df['helio_sep_M_deg'].iloc[i]:4.1f}°")
        if planet_enabled["Saturn"]:
            helio_bits.append(f"S: {df['helio_sep_S_deg'].iloc[i]:4.1f}°")
        if helio_bits:
            parts.append("Helio sep " + " ".join(helio_bits))

        glare_bits = []
        if planet_enabled["Mars"] and float(df["mars_elong_deg"].iloc[i]) < 15:
            glare_bits.append("Mars elong < 15°")
        if planet_enabled["Jupiter"] and float(df["jup_elong_deg"].iloc[i]) < 15:
            glare_bits.append("Jupiter elong < 15°")
        if planet_enabled["Saturn"] and float(df["sat_elong_deg"].iloc[i]) < 15:
            glare_bits.append("Saturn elong < 15°")
        if glare_bits:
            parts.append("Sun glare: " + ", ".join(glare_bits) + " (hard/unobservable)")

        return "\n".join(parts)

    txt = ax1.text(
        0.98, 0.98, make_readout_text(i0),
        transform=ax1.transAxes, va="top", ha="right",
        fontsize=9, family="monospace", color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.85, lw=0.5, edgecolor="white")
    )

    # ---------- FIG 2: Range + Elongation ----------
    fig2, ax2 = plt.subplots(figsize=(12.5, 4.8), num=f"{APP_NAME} {APP_VERSION} — Ranges & Elongation")
    fig2.patch.set_facecolor(BACKGROUND_COLOR)
    ax2.set_facecolor(BACKGROUND_COLOR)

    # Secondary axis for elongation
    ax2b = ax2.twinx()
    ax2b.set_facecolor(BACKGROUND_COLOR)

    # Ranges
    if planet_enabled["Jupiter"]:
        ax2.plot(days, df["earth_jupiter_range_AU"], lw=1.8, alpha=0.9, label="Earth–Jupiter range (AU)")
    if planet_enabled["Mars"]:
        ax2.plot(days, df["earth_mars_range_AU"], lw=1.6, alpha=0.9, label="Earth–Mars range (AU)")
    if planet_enabled["Saturn"]:
        ax2.plot(days, df["earth_saturn_range_AU"], lw=1.6, alpha=0.9, label="Earth–Saturn range (AU)")

    # Elongations (simple single-line per planet; keep your banding if you want later)
    if planet_enabled["Jupiter"]:
        ax2b.plot(days, df["jup_elong_deg"], lw=1.4, alpha=0.9, label="Jupiter elongation (deg)")
    if planet_enabled["Mars"]:
        ax2b.plot(days, df["mars_elong_deg"], lw=1.2, alpha=0.9, label="Mars elongation (deg)")
    if planet_enabled["Saturn"]:
        ax2b.plot(days, df["sat_elong_deg"], lw=1.2, alpha=0.9, label="Saturn elongation (deg)")

    ax2.set_xlabel("Date", color="white")
    ax2.set_ylabel("Range (AU)", color="white")
    ax2b.set_ylabel("Elongation (deg)", color="white")

    ax2.tick_params(axis="x", colors="white")
    ax2.tick_params(axis="y", colors="white")
    ax2b.tick_params(axis="y", colors="white")
    ax2.grid(True, ls=":", alpha=0.25, color="white")

    # Solar glare shading (any enabled planet under DIFFICULT threshold)
    if SHOW_SOLAR_GLARE_MASK:
        glare_mask = np.zeros(len(df), dtype=bool)
        if planet_enabled["Mars"]:
            glare_mask |= (df["mars_elong_deg"].to_numpy() < ELONG_DIFFICULT_DEG)
        if planet_enabled["Jupiter"]:
            glare_mask |= (df["jup_elong_deg"].to_numpy() < ELONG_DIFFICULT_DEG)
        if planet_enabled["Saturn"]:
            glare_mask |= (df["sat_elong_deg"].to_numpy() < ELONG_DIFFICULT_DEG)

        glare_spans = _mask_to_spans(glare_mask)
        for a, b in glare_spans:
            ax2.axvspan(days.iloc[a], days.iloc[b], color=GLARE_SHADE_COLOR, alpha=GLARE_SHADE_ALPHA, zorder=0)
        ax2b.axhline(float(GLARE_SHADE_BELOW_DEG), color=GLARE_SHADE_COLOR, lw=1.0, alpha=0.35)

    # Vertical time cursor
    vline = ax2.axvline(days.iloc[i0], ls=":", color="white", alpha=0.9)

    # Legend combined
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right",
               frameon=True, facecolor="black", edgecolor="white",
               fontsize=8, labelcolor="white")

    # ---------- Slider + Reset ----------
    slider_ax = fig1.add_axes([0.12, 0.06, 0.76, 0.03], facecolor="#222")
    slider = Slider(slider_ax, "", 0, MAX_IDX, valinit=i0, valstep=1)
    slider.valtext.set_color("white")

    btn_ax = fig1.add_axes([0.90, 0.05, 0.08, 0.05])
    btn = Button(btn_ax, "Reset")

    def update(val):
        i = int(val)
        i = max(0, min(MAX_IDX, i))

        # Markers
        E.set_offsets([[df["x_E"].iloc[i], df["y_E"].iloc[i]]])
        M.set_offsets([[df["x_M"].iloc[i], df["y_M"].iloc[i]]])
        J.set_offsets([[df["x_J"].iloc[i], df["y_J"].iloc[i]]])
        S.set_offsets([[df["x_S"].iloc[i], df["y_S"].iloc[i]]])

        # Connectors
        lineEM.set_data([df["x_E"].iloc[i], df["x_M"].iloc[i]], [df["y_E"].iloc[i], df["y_M"].iloc[i]])
        lineEJ.set_data([df["x_E"].iloc[i], df["x_J"].iloc[i]], [df["y_E"].iloc[i], df["y_J"].iloc[i]])
        lineES.set_data([df["x_E"].iloc[i], df["x_S"].iloc[i]], [df["y_E"].iloc[i], df["y_S"].iloc[i]])

        if SHOW_SOLAR_GLARE_MASK:
            style_connector(lineEJ, float(df["jup_elong_deg"].iloc[i]))
            style_connector(lineEM, float(df["mars_elong_deg"].iloc[i]))
            style_connector(lineES, float(df["sat_elong_deg"].iloc[i]))

        # Readout + view limits + legend
        txt.set_text(make_readout_text(i))
        apply_planet_visibility()
        set_view_limits()

        # Fig2 cursor
        vline.set_xdata([days.iloc[i], days.iloc[i]])

        fig1.canvas.draw_idle()
        fig2.canvas.draw_idle()

    slider.on_changed(update)

    def reset_view(_):
        slider.set_val(i0)

    btn.on_clicked(reset_view)

    # Keyboard stepping
    def key_press(event):
        current_val = int(slider.val)
        if event.key == "right":
            slider.set_val(min(MAX_IDX, current_val + 1))
        elif event.key == "left":
            slider.set_val(max(0, current_val - 1))

    fig1.canvas.mpl_connect("key_press_event", key_press)
    fig2.canvas.mpl_connect("key_press_event", key_press)

    # Apply visibility once on startup
    apply_planet_visibility()

    plt.show()


if __name__ == "__main__":
    main()