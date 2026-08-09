from config import *


from plotting import (set_view_limits,
                      refresh_legend, 
                      make_readout_text)



def apply_planet_visibility(planet_artists, planet_enabled, ax1):
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


def update(
    val,
    MAX_IDX,
    df,
    dates,
    days,
    E, M, J, S,
    lineEM, lineEJ, lineES,
    txt,
    vline,
    fig1,
    fig2,
    ax1,
    planet_enabled,
    style_connector,
    planet_artists
):


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
    txt.set_text(make_readout_text(i, dates, planet_enabled, df))
    apply_planet_visibility(planet_artists, planet_enabled, ax1)
    set_view_limits(ax1,planet_enabled)

    # Fig2 cursor
    vline.set_xdata([days.iloc[i], days.iloc[i]])

    fig1.canvas.draw_idle()
    fig2.canvas.draw_idle()
