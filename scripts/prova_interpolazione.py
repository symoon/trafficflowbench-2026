"""Confronto rapido sul train: baseline ufficiale, interpolazione nel tempo, nello spazio, miscela."""
import sys

import numpy as np
import pandas as pd

from tfb.dati import RADICE_PROGETTO, bersagli_task1, tensori
from tfb.metrica import s_lwr_stima, s_state


def interpola_asse(x: np.ndarray, asse: int) -> np.ndarray:
    """Interpolazione lineare dei NaN lungo un asse; agli estremi ripete il valore più vicino."""
    x = np.moveaxis(x, asse, -1)
    forma = x.shape
    y = x.reshape(-1, forma[-1]).copy()
    idx = np.arange(forma[-1])
    for i in range(y.shape[0]):
        r = y[i]
        ok = ~np.isnan(r)
        if ok.sum() == 0 or ok.all():
            continue
        r[~ok] = np.interp(idx[~ok], idx[ok], r[ok])
    return np.moveaxis(y.reshape(forma), -1, asse)


p = sys.argv[1] if len(sys.argv) > 1 else "D12_I5_N"
vero = tensori(p, "train", "vero")
masc = tensori(p, "train", "mascherato")
b = bersagli_task1(p, "train", masc)
g, t, l = b.g.to_numpy(), b.t.to_numpy(), b.l.to_numpy()


def riporta(nome, v, q):
    s = s_state(b, vero, v, q)
    w = s_lwr_stima(b, vero, v, q)
    print(f"{nome:28s} S_state {s['S_state']:.4f}  " + "  ".join(
        f"{r}: v {s[r]['rmse_v']:.2f} q {s[r]['rmse_q']:.1f}" for r in ("R1", "R2", "R3")) + f"  S_LWR~ {w['S_LWR']:.3f}", flush=True)


uff = pd.read_csv(RADICE_PROGETTO / "outputs" / "baseline" / f"state_train_{p}.csv", dtype={"timestamp": str, "station_id": str, "link_id": str})
uff = b[["timestamp", "station_id", "link_id", "mask_regime"]].merge(uff, on=["timestamp", "station_id", "link_id", "mask_regime"], how="left", validate="one_to_one")
riporta("baseline ufficiale", uff.speed_kmh.to_numpy(), uff.flow_vph.to_numpy())

# Solo le celle osservate ed eleggibili del giorno sono evidenza; le non eleggibili hanno pct_observed basso.
ev = masc.eleggibile & ~np.isnan(masc.speed)
vs = np.where(ev, masc.speed, np.nan)
qs = np.where(ev, masc.flow, np.nan)
vt, qt = interpola_asse(vs, 1), interpola_asse(qs, 1)
riporta("tempo, stesso tratto", vt[g, t, l], qt[g, t, l])
vx, qx = interpola_asse(vs, 2), interpola_asse(qs, 2)
riporta("spazio, stesso istante", vx[g, t, l], qx[g, t, l])
riporta("media tempo+spazio", (vt[g, t, l] + vx[g, t, l]) / 2, (qt[g, t, l] + qx[g, t, l]) / 2)
riporta("perfetto", vero.speed[g, t, l], vero.flow[g, t, l])
