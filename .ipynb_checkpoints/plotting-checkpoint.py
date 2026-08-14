from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from config import *

import sys
import matplotlib
import numpy as np
import pandas as pd
from astropy.time import Time

# Helper functions

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


def refresh_legend(ax):
    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.remove()
        except Exception:
            pass

    handles, labels = ax.get_legend_handles_labels()

    items = []
    seen = set()

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
        ax.legend(hs, ls, loc="lower right", frameon=True, fontsize=8)


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

def make_readout_text(i, dates, planet_enabled, df ):
    d = dates.iloc[i].strftime("%d-%b-%Y")
    parts = [f"Date: {d}"]


    if planet_enabled["Venus"]:
        parts.append(f"Earth–Venus:   {df['earth_venus_range_AU'].iloc[i]:6.3f} AU   Elong: {df['venus_elong_deg'].iloc[i]:5.1f}°")

    if planet_enabled["Jupiter"]:
        parts.append(f"Earth–Jupiter: {df['earth_jupiter_range_AU'].iloc[i]:6.3f} AU   Elong: {df['jup_elong_deg'].iloc[i]:5.1f}°")
    if planet_enabled["Mars"]:
        parts.append(f"Earth–Mars:   {df['earth_mars_range_AU'].iloc[i]:6.3f} AU   Elong: {df['mars_elong_deg'].iloc[i]:5.1f}°")
    if planet_enabled["Saturn"]:
        parts.append(f"Earth–Saturn: {df['earth_saturn_range_AU'].iloc[i]:6.3f} AU   Elong: {df['sat_elong_deg'].iloc[i]:5.1f}°")

    helio_bits = []

    if planet_enabled["Venus"]:
        helio_bits.append(f"V: {df['helio_sep_V_deg'].iloc[i]:4.1f}°")


    if planet_enabled["Jupiter"]:
        helio_bits.append(f"J: {df['helio_sep_J_deg'].iloc[i]:4.1f}°")
    if planet_enabled["Mars"]:
        helio_bits.append(f"M: {df['helio_sep_M_deg'].iloc[i]:4.1f}°")
    if planet_enabled["Saturn"]:
        helio_bits.append(f"S: {df['helio_sep_S_deg'].iloc[i]:4.1f}°")
    if helio_bits:
        parts.append("Helio sep " + " ".join(helio_bits))

    glare_bits = []

    if planet_enabled["Venus"] and float(df["venus_elong_deg"].iloc[i]) < 15:
        glare_bits.append("Venus elong < 15°")


    if planet_enabled["Mars"] and float(df["mars_elong_deg"].iloc[i]) < 15:
        glare_bits.append("Mars elong < 15°")
    if planet_enabled["Jupiter"] and float(df["jup_elong_deg"].iloc[i]) < 15:
        glare_bits.append("Jupiter elong < 15°")
    if planet_enabled["Saturn"] and float(df["sat_elong_deg"].iloc[i]) < 15:
        glare_bits.append("Saturn elong < 15°")
    if glare_bits:
        parts.append("Sun glare: " + ", ".join(glare_bits) + " (hard/unobservable)")

    return "\n".join(parts)

# Orbit drawings

def draw_guide_orbits(ax, theta, planet_enabled, planet_artists):
    """
    Draw the reference circular orbits for the enabled planets.
    """

    ax.plot(
        np.cos(theta),
        np.sin(theta),
        "--",
        color=GUIDE_COLOR,
        alpha=0.45,
        lw=1.0,
        label="Earth orbit (1 AU)",
    )

    if planet_enabled["Venus"]:
        h = ax.plot(
            VENUS_ORBIT_AU * np.cos(theta),
            VENUS_ORBIT_AU * np.sin(theta),
            "--",
            color=GUIDE_COLOR,
            alpha=0.35,
            lw=1.0,
            label=f"Venus orbit ({VENUS_ORBIT_AU:.3f} AU)",
        )[0]
        register_artists(planet_artists,"Venus", h)

    if planet_enabled["Mars"]:
        h = ax.plot(
            MARS_ORBIT_AU * np.cos(theta),
            MARS_ORBIT_AU * np.sin(theta),
            "--",
            color=GUIDE_COLOR,
            alpha=0.35,
            lw=1.0,
            label=f"Mars orbit ({MARS_ORBIT_AU:.3f} AU)",
        )[0]
        register_artists(planet_artists,"Mars", h)

    if planet_enabled["Jupiter"]:
        h = ax.plot(
            JUPITER_ORBIT_AU * np.cos(theta),
            JUPITER_ORBIT_AU * np.sin(theta),
            "--",
            color=GUIDE_COLOR,
            alpha=0.28,
            lw=1.0,
            label=f"Jupiter orbit ({JUPITER_ORBIT_AU:.3f} AU)",
        )[0]
        register_artists(planet_artists, "Jupiter", h)

    if planet_enabled["Saturn"]:
        h = ax.plot(
            SATURN_ORBIT_AU * np.cos(theta),
            SATURN_ORBIT_AU * np.sin(theta),
            "--",
            color=GUIDE_COLOR,
            alpha=0.20,
            lw=1.0,
            label=f"Saturn orbit ({SATURN_ORBIT_AU:.3f} AU)",
        )[0]
        register_artists(planet_artists,"Saturn", h)

def draw_sun(ax):
    
    ax.scatter([0], [0], s=SUN_GLOW_SIZE, color=SUN_GLOW_COLOR, alpha=0.22, zorder=3)
    ax.scatter([0], [0], s=SUN_CORE_SIZE, color=SUN_CORE_COLOR, edgecolor="white", label="Sun", zorder=4)




def draw_planet_tracks(ax, df, planet_enabled, planet_artists):

    ax.plot(df["x_E"], df["y_E"], color=EARTH_COLOR, lw=1.5, alpha=0.55, label="Earth (HORIZONS)")




    venus_track = ax.plot(df["x_V"], df["y_V"], color=VENUS_COLOR, lw=1.8, alpha=0.50, label="Venus (HORIZONS)")[0]

    mars_track = ax.plot(df["x_M"], df["y_M"], color=MARS_COLOR, lw=1.8, alpha=0.50, label="Mars (HORIZONS)")[0]
    jup_track  = ax.plot(df["x_J"], df["y_J"], color=JUPITER_COLOR, lw=2.0, alpha=0.55, label="Jupiter (HORIZONS)")[0]
    sat_track  = ax.plot(df["x_S"], df["y_S"], color=SATURN_COLOR, lw=2.0, alpha=0.45, label="Saturn (HORIZONS)")[0]


    register_artists(planet_artists,"Venus", venus_track)
    
    register_artists(planet_artists,"Mars", mars_track)
    register_artists(planet_artists,"Jupiter", jup_track)
    register_artists(planet_artists,"Saturn", sat_track)





def draw_current_markers(ax, df, i0, planet_artists):
    E = ax.scatter([df["x_E"].iloc[i0]], [df["y_E"].iloc[i0]], s=EARTH_SIZE, color=EARTH_COLOR, edgecolor="white",
                    label="_nolegend_", zorder=6)

    V = ax.scatter([df["x_V"].iloc[i0]], [df["y_V"].iloc[i0]], s=VENUS_SIZE, color=VENUS_COLOR, edgecolor="white",
                    label="_nolegend_", zorder=6)

    
    M = ax.scatter([df["x_M"].iloc[i0]], [df["y_M"].iloc[i0]], s=MARS_SIZE, color=MARS_COLOR, edgecolor="white",
                    label="_nolegend_", zorder=6)
    J = ax.scatter([df["x_J"].iloc[i0]], [df["y_J"].iloc[i0]], s=JUPITER_SIZE, color=JUPITER_COLOR, edgecolor="white",
                    label="_nolegend_", zorder=6)
    S = ax.scatter([df["x_S"].iloc[i0]], [df["y_S"].iloc[i0]], s=SATURN_SIZE, color=SATURN_COLOR, edgecolor="white",
                    label="_nolegend_", zorder=6)


    register_artists(planet_artists,"Venus", V)


    register_artists(planet_artists,"Mars", M)
    register_artists(planet_artists,"Jupiter", J)
    register_artists(planet_artists,"Saturn", S)

    return V, E, M, J, S

# Event plotting

def draw_events(ax, df, planet_artists, planet: str, xcol: str, ycol: str, opp_idxs, conj_idxs, size=EVENT_SIZE, fs=8):
    # Opposition
    for k, ii in enumerate(opp_idxs):
        h = ax.scatter(df[xcol].iloc[ii], df[ycol].iloc[ii],
                        s=size, facecolor="none", edgecolor=EVENT_OPP_COLOR, lw=EVENT_MARKER_LW, zorder=9,
                        label="Opposition" if (planet == "Jupiter" and k == 0) else "_nolegend_")
        register_artists(planet_artists,planet, h)
        if SHOW_EVENT_LABELS:
            yoff = 0.18 if (k % 2 == 0) else 0.30
            t = ax.text(df[xcol].iloc[ii], df[ycol].iloc[ii] + yoff,
                            f"Opp\n{df['t'].iloc[ii].strftime(DATE_FMT)}",
                            ha="center", va="bottom", fontsize=fs, color="white",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, lw=0))
            register_artists(planet_artists,planet, t)

    # Conjunction
    for k, ii in enumerate(conj_idxs):
        h = ax.scatter(df[xcol].iloc[ii], df[ycol].iloc[ii],
                        s=size, facecolor="none", edgecolor=EVENT_CONJ_COLOR, lw=EVENT_MARKER_LW, zorder=9,
                        label="Conjunction" if (planet == "Jupiter" and k == 0) else "_nolegend_")
        register_artists(planet_artists,planet, h)
        if SHOW_EVENT_LABELS:
            yoff = -0.22 if (k % 2 == 0) else -0.36
            t = ax.text(df[xcol].iloc[ii], df[ycol].iloc[ii] + yoff,
                            f"Conj\n{df['t'].iloc[ii].strftime(DATE_FMT)}",
                            ha="center", va="top", fontsize=fs, color="white",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, lw=0))
            register_artists(planet_artists,planet, t)


# Connector plotting

def draw_connectors(ax, df, planet_artists, i0):


    lineEV = ax.plot([df["x_E"].iloc[i0], df["x_V"].iloc[i0]], [df["y_E"].iloc[i0], df["y_V"].iloc[i0]],
                      color="white", alpha=0.55, lw=CONNECTOR_LW_USABLE, zorder=5)[0]


    lineEM = ax.plot([df["x_E"].iloc[i0], df["x_M"].iloc[i0]], [df["y_E"].iloc[i0], df["y_M"].iloc[i0]],
                      color="white", alpha=0.55, lw=CONNECTOR_LW_USABLE, zorder=5)[0]
    lineEJ = ax.plot([df["x_E"].iloc[i0], df["x_J"].iloc[i0]], [df["y_E"].iloc[i0], df["y_J"].iloc[i0]],
                      color="white", alpha=0.65, lw=CONNECTOR_LW_USABLE, zorder=5)[0]
    lineES = ax.plot([df["x_E"].iloc[i0], df["x_S"].iloc[i0]], [df["y_E"].iloc[i0], df["y_S"].iloc[i0]],
                      color="white", alpha=0.45, lw=CONNECTOR_LW_USABLE, zorder=5)[0]


    register_artists(planet_artists,"Venus", lineEV)
    
    register_artists(planet_artists,"Mars", lineEM)
    register_artists(planet_artists,"Jupiter", lineEJ)
    register_artists(planet_artists,"Saturn", lineES)

    return lineEV, lineEM, lineEJ, lineES


#Figure 2 construction
#def build_figure2(planet_enabled, days, df):

def build_analysis_figure(planet_enabled, days, df):

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

    return fig2, ax2, ax2b

def set_view_limits(ax1, planet_enabled):
    lim = compute_view_lim(planet_enabled)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)

def register_artists(planet_artists, planet, *artists):
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