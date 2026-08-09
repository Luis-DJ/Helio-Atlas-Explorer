#!/usr/bin/env python3
"""
HelioAtlas Jupiter/Mars/Saturn opposition/conjunction visualiser (interactive)
Classic Ecliptic View + Dynamic View Limits + Per-planet visibility control

Fixes included:
- Restores full Fig 2 + slider UI (ax2/ax2b/etc. are defined).
- Fixes : Mars event markers were incorrectly registered to Jupiter.
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

from config import *

from geometry import (
    ensure_utc,
    horizons_times_to_utc,
    angle_deg_vectorized

)

from horizons import *

from events import (
    find_conj_opp_events
)

from plotting import (
    _mask_to_spans,
    maximize_figure,
    refresh_legend,
    style_connector,
    make_readout_text,
    draw_guide_orbits,
    draw_sun,
    draw_planet_tracks,
    draw_current_markers,
    draw_events,
    draw_connectors,
    compute_view_lim,
    build_analysis_figure)

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
    fig1, ax1 = plt.subplots(figsize=(12.5, 9.5), num=f"{APP_NAME} {APP_VERSION} — Ranges & Elongation")
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

    refresh_legend(ax1)

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
        refresh_legend(ax1)

    def set_view_limits():
        lim = compute_view_lim(planet_enabled)
        ax1.set_xlim(-lim, lim)
        ax1.set_ylim(-lim, lim)
     

    theta = np.linspace(0, 2*np.pi, 1200)

    # Guide orbits (only for enabled planets)
    
    draw_guide_orbits(ax1, theta, planet_enabled, _reg)

    # Sun

    draw_sun(ax1)
    
    # Tracks

    draw_planet_tracks(ax1, df, _reg)

    # Current markers
    E, M, J, S = draw_current_markers(ax1, df, i0, _reg)
    
    # Connectors

    lineEM, lineEJ, lineES = draw_connectors(ax1, df, _reg, i0)

    _reg("Mars", lineEM)
    _reg("Jupiter", lineEJ)
    _reg("Saturn", lineES)

    if SHOW_SOLAR_GLARE_MASK:
        style_connector(lineEJ, float(df["jup_elong_deg"].iloc[i0]))
        style_connector(lineEM, float(df["mars_elong_deg"].iloc[i0]))
        style_connector(lineES, float(df["sat_elong_deg"].iloc[i0]))

    # Events (rings + optional labels), all registered to the correct planet
                
    # Jupiter full-size; Mars slightly smaller; Saturn full-size
    draw_events(ax1, df, _reg, "Jupiter", "x_J", "y_J", j_opp, j_conj, size=EVENT_SIZE, fs=8)
    draw_events(ax1, df, _reg, "Mars",    "x_M", "y_M", m_opp, m_conj, size=EVENT_SIZE*0.85, fs=7.5)
    draw_events(ax1, df, _reg, "Saturn",  "x_S", "y_S", s_opp, s_conj, size=EVENT_SIZE, fs=8)

    ax1.set_aspect("equal")

    set_view_limits()

    ax1.set_xlabel("x (AU)", color="white")
    ax1.set_ylabel("y (AU)", color="white")
    ax1.tick_params(axis="x", colors="white")
    ax1.tick_params(axis="y", colors="white")
    ax1.set_title(f"Earth–Planet Geometry ({APP_VERSION}) — Heliocentric ecliptic", color="white")
    ax1.grid(True, ls=":", alpha=0.35, color="white")

    txt = ax1.text(
        0.98, 0.98, make_readout_text(i0, dates, planet_enabled, df),
        transform=ax1.transAxes, va="top", ha="right",
        fontsize=9, family="monospace", color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.85, lw=0.5, edgecolor="white")
    )

    # ---------- FIG 2: Range + Elongation ----------

    fig2, ax2, ax2b = build_analysis_figure(planet_enabled, days, df)

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