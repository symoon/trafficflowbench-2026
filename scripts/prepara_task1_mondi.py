"""Task 1: le stesse celle di prepara_task1.py in mondi spostati, per allenare con randomizzazione del mondo.

Per ogni corridoio, blocco e copia si estrae un mondo: a ogni tratto uno scarto di velocità
δ ~ N(0; 1,2 km/h), pieno in flusso libero e nullo in coda (cv_mondo.sposta), applicato allo
strato mascherato e alla verità. Le caratteristiche si ricalcolano sul mondo spostato; i flussi
restano quelli veri. Ogni blocco ha i suoi mondi, così un modello allenato su più blocchi vede
più mondi e non può imparare i livelli di velocità del singolo tratto.

    uv run python scripts/prepara_task1_mondi.py [--copie 1]
"""
import argparse
import time
import zlib

import numpy as np
import pandas as pd

from cv_mondo import FRAZIONE, SD_MONDO, sposta
from tfb.caratteristiche import calcola
from tfb.dati import RADICE_PROGETTO, pannelli, tensori, tratti

ENTRATA = RADICE_PROGETTO / "outputs" / "task1" / "caratteristiche"
USCITA = RADICE_PROGETTO / "outputs" / "task1" / "caratteristiche_mondi"
CELLE = ["g", "t", "l", "blocco", "completo", "mask_regime", "timestamp", "station_id", "link_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copie", type=int, default=1)
    a = ap.parse_args()
    USCITA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for p in pannelli():
        celle = pd.read_parquet(ENTRATA / f"{p}.parquet", columns=CELLE)
        masc, vero = tensori(p, "train"), tensori(p, "train", "vero")
        tr = tratti(p)
        vf, corsie = tr.free_speed_kmh.to_numpy(np.float32), tr.lanes.to_numpy(np.float32)
        rif = np.where(np.isfinite(vero.speed), vero.speed, np.nanmedian(vero.speed))
        pezzi = []
        for copia in range(a.copie):
            for k in range(3):
                rng = np.random.default_rng([copia, k, zlib.crc32(p.encode())])
                delta = rng.normal(0.0, SD_MONDO, len(vf)).astype(np.float32)
                m, v = sposta(masc, vf, delta, rif), sposta(vero, vf, delta, rif)
                c = celle[celle.blocco == k].reset_index(drop=True)
                X = calcola(m, c, FRAZIONE)
                g, t, l = c.g.to_numpy(), c.t.to_numpy(), c.l.to_numpy()
                X["y_v"] = v.speed[g, t, l]
                X["y_q"] = vero.flow[g, t, l] / corsie[l]
                for col in CELLE:
                    X[col] = c[col].to_numpy()
                X["pannello"], X["copia"] = p, copia
                pezzi.append(X)
        pd.concat(pezzi, ignore_index=True).to_parquet(USCITA / f"{p}.parquet", index=False)
        print(f"{p}: {sum(len(x) for x in pezzi):,} celle ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
