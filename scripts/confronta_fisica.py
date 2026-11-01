"""Scarto di S_physics tra due invii (complemento di confronta_invio.py, che copre S_state, S_queue e S_ODME).

S_FD si calcola esatto anche senza verità (metrica.s_fd_valutato). Per S_LWR la verità serve al
denominatore: |ΔS_LWR| <= Σ|dN_a - dN_b| / Σ|dN| sulle transizioni eleggibili, con Σ|dN| preso
dall'invio di riferimento, quindi è una stima e non un limite esatto. Si leggono gli stati salvati da
genera_invio.py (outputs/invii/<versione>/stato_*.npz).

    .venv/bin/python scripts/confronta_fisica.py <cartella invio nuovo> <cartella invio di riferimento>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tfb.dati import bersagli_task1, pannelli, tensori
from tfb.invio import SPLIT_VALUTATI
from tfb.metrica import parametri_fisica, s_fd_valutato

A, B = Path(sys.argv[1]), Path(sys.argv[2])
righe = []
for p in pannelli():
    L = parametri_fisica(p).length_km.to_numpy(float)
    for s in SPLIT_VALUTATI:
        m = tensori(p, s)
        b = bersagli_task1(p, s, m)
        g, t, l = b.g.to_numpy(), b.t.to_numpy(), b.l.to_numpy()
        za, zb = np.load(A / f"stato_{p}_{s}.npz"), np.load(B / f"stato_{p}_{s}.npz")
        fa = s_fd_valutato(b, m, za["v"][g, t, l], za["q"][g, t, l])["S_FD"]
        fb = s_fd_valutato(b, m, zb["v"][g, t, l], zb["q"][g, t, l])["S_FD"]
        N = {}
        for k, z in (("a", za), ("b", zb)):
            v, q = m.speed.astype(float), m.flow.astype(float)
            v[g, t, l], q[g, t, l] = z["v"][g, t, l], z["q"][g, t, l]
            N[k] = q / np.maximum(v, 1.0) * L
        ele = m.eleggibile & np.isfinite(N["b"]) & np.isfinite(N["a"])
        mm = ele[:, 1:] & ele[:, :-1]
        dA, dB = np.diff(N["a"], axis=1), np.diff(N["b"], axis=1)
        lwr = np.abs(dA - dB)[mm].sum() / (np.abs(dB)[mm].sum() + 1e-9)
        righe.append(dict(pannello=p, split=s, dFD=abs(fa - fb), dLWR=lwr))
    print(p, flush=True)
tab = pd.DataFrame(righe)
tab["corridoio"] = tab.pannello.str.rsplit("_", n=1).str[0]
agg = tab.groupby(["split", "corridoio"])[["dFD", "dLWR"]].mean().groupby("split").mean()
agg["dS_physics"] = agg.dFD / 3 + 2 * agg.dLWR / 3
agg["sul_totale"] = 0.15 * agg.dS_physics
pd.set_option("display.width", 200)
print(tab.round(6).to_string())
print(agg.to_string())
