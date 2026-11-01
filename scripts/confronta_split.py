"""Statistiche osservabili per mese: train, validation e private sullo strato mascherato.

Serve a vedere se validation e private assomigliano al train (stessa domanda, stessa
capacità, stessa frequenza di code) prima di fidarsi di un modello allenato sul train.
"""
import numpy as np
import pandas as pd

from tfb.dati import SPLIT, pannelli, tensori, tratti
from tfb.task1 import evidenza

righe = []
for p in pannelli():
    tr = tratti(p)
    vf = tr.free_speed_kmh.to_numpy(float)
    corsie = tr.lanes.to_numpy(float)
    for s in SPLIT:
        t = tensori(p, s)
        v, q = evidenza(t)
        mesi = pd.Series(t.date).str.slice(0, 7).to_numpy()
        for m in np.unique(mesi):
            sel = mesi == m
            vv, qq = v[sel], q[sel] / corsie
            ok = np.isfinite(vv)
            coda = (vv <= 0.6 * vf)[ok]
            picco = qq[:, 7 * 12:9 * 12][np.isfinite(qq[:, 7 * 12:9 * 12])]
            righe.append(dict(pannello=p, split=s, mese=m, giorni=int(sel.sum()),
                              v_media=float(np.nanmean(vv)), quota_coda=float(coda.mean()),
                              q_corsia_media=float(np.nanmean(qq)), q_corsia_p99=float(np.nanpercentile(qq, 99)),
                              q_picco_7_9=float(np.mean(picco)), copertura=float(ok.mean())))
d = pd.DataFrame(righe)
pd.set_option("display.width", 200)
print(d.groupby(["split", "mese"])[["v_media", "quota_coda", "q_corsia_media", "q_corsia_p99", "q_picco_7_9", "copertura"]].mean().round(3).to_string())
piv = d.pivot_table(index="pannello", columns="split", values="quota_coda", aggfunc="mean")
print("\nquota di celle in coda per corridoio\n", piv.round(4).to_string())
piv = d.pivot_table(index="pannello", columns="split", values="q_corsia_p99", aggfunc="mean")
print("\nflusso per corsia, 99° percentile\n", piv.round(0).to_string())
d.to_csv("outputs/confronto_split.csv", index=False)
