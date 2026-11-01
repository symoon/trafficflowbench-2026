"""Task 1: estrae una volta le caratteristiche per la validazione a blocchi temporali.

Il train (273 giorni) si divide in tre blocchi di tre mesi. Per ogni corridoio e
blocco si tengono GIORNI_COMPLETI giorni con tutte le celle bersaglio (servono alla
stima di S_LWR, che guarda le transizioni tra istanti consecutivi) e un campione
casuale di celle dagli altri giorni. Le caratteristiche usano solo il giorno della
cella, quindi i blocchi non si contaminano.

    uv run python scripts/prepara_task1.py
"""
import time

import numpy as np
import pandas as pd

from tfb.caratteristiche import calcola
from tfb.dati import RADICE_PROGETTO, bersagli_task1, pannelli, tensori, tratti

FRAZIONE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
CONFINI = (0, 92, 183, 273)  # giu–ago, set–nov, dic–feb
GIORNI_COMPLETI = 8
CAMPIONE = 80_000
USCITA = RADICE_PROGETTO / "outputs" / "task1" / "caratteristiche"


def main() -> None:
    USCITA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for p in pannelli():
        vero, masc = tensori(p, "train", "vero"), tensori(p, "train", "mascherato")
        b = bersagli_task1(p, "train", masc)
        corsie = tratti(p).lanes.to_numpy(np.float32)
        pezzi = []
        for k in range(3):
            giorni = np.arange(CONFINI[k], CONFINI[k + 1])
            # giorni completi stratificati per regime, così ogni regime ha la sua stima
            completi = []
            for r in ("R1", "R2", "R3"):
                cand = giorni[masc.regime[giorni] == r]
                completi += list(rng.choice(cand, size=min(GIORNI_COMPLETI // 3 + 1, len(cand)), replace=False))
            completi = np.array(sorted(completi))
            nel_blocco = b.g.between(CONFINI[k], CONFINI[k + 1] - 1).to_numpy()
            pieni = nel_blocco & np.isin(b.g.to_numpy(), completi)
            altri = np.flatnonzero(nel_blocco & ~pieni)
            scelti = np.sort(rng.choice(altri, size=min(CAMPIONE, len(altri)), replace=False))
            c = pd.concat([b[pieni].assign(completo=True), b.iloc[scelti].assign(completo=False)], ignore_index=True)
            c["blocco"] = k
            pezzi.append(c)
        celle = pd.concat(pezzi, ignore_index=True)
        X = calcola(masc, celle, FRAZIONE)
        g, t, l = celle.g.to_numpy(), celle.t.to_numpy(), celle.l.to_numpy()
        X["y_v"] = vero.speed[g, t, l]
        X["y_q"] = vero.flow[g, t, l] / corsie[l]
        for c in ("g", "t", "l", "blocco", "completo", "mask_regime", "timestamp", "station_id", "link_id"):
            X[c] = celle[c].to_numpy()
        X["pannello"] = p
        X.to_parquet(USCITA / f"{p}.parquet", index=False)
        print(f"{p}: {len(X):,} celle, di cui {int(celle.completo.sum()):,} in giorni completi ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
