
import numpy as np
import pandas as pd

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


def _fill_zero_signs(s: np.ndarray) -> np.ndarray:
    s = s.astype(float)
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i-1]
    for i in range(len(s)-2, -1, -1):
        if s[i] == 0:
            s[i] = s[i+1]
    return s
