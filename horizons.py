import numpy as np
import pandas as pd

from astroquery.jplhorizons import Horizons

from astropy.time import Time

from config import *
from geometry import (
    ensure_utc,
    horizons_times_to_utc,
    angle_deg_vectorized,
)




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
