"""Task 1: salva le previsioni fuori campione della validazione a blocchi sui giorni completi.

Servono a provare nuove stime della parte fisica senza riallenare: per ogni blocco i modelli
si allenano sugli altri due (come cv_mondo.py, mondo del train) e si salvano velocità e
flusso totale previsti nelle celle bersaglio dei giorni completi, per l'interpolazione e per
LightGBM a 800 e 3200 alberi.

    uv run python scripts/cv_previsioni.py [--allenamento train|entrambi] [--thread 16]
"""
import argparse
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from cv_mondo import ALBERI
from cv_task1 import CARTELLA, META, PARAMETRI
from tfb.caratteristiche import base
from tfb.dati import pannelli, tratti

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allenamento", choices=["train", "entrambi"], default="train")
    ap.add_argument("--thread", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    d = pd.concat([pd.read_parquet(CARTELLA / "caratteristiche" / f"{p}.parquet") for p in pannelli()], ignore_index=True)
    colonne = [c for c in d.columns if c not in META]
    fonti = [d]
    if a.allenamento == "entrambi":
        fonti.append(pd.concat([pd.read_parquet(CARTELLA / "caratteristiche_mondi" / f"{p}.parquet") for p in pannelli()],
                               ignore_index=True).drop(columns="copia"))
    A = pd.concat(fonti, ignore_index=True)
    bv_a, bq_a = base(A, "v"), base(A, "q")
    pezzi = []
    for k in range(3):
        al = (A.blocco != k).to_numpy()
        modelli = {}
        for nome, y, b_ in (("v", A.y_v.to_numpy(), bv_a), ("q", A.y_q.to_numpy(), bq_a)):
            r = y - b_
            ok = al & np.isfinite(r)
            modelli[nome] = lgb.train(dict(PARAMETRI, num_threads=a.thread), lgb.Dataset(A.loc[ok, colonne], r[ok]),
                                      num_boost_round=max(ALBERI))
        print(f"fold {k}: modelli ({time.time() - t0:.0f} s)", flush=True)
        for p in pannelli():
            X = d[(d.pannello == p) & (d.blocco == k) & d.completo].reset_index(drop=True)
            corsie = tratti(p).lanes.to_numpy(np.float32)[X.l.to_numpy()]
            xv, xq = base(X, "v"), base(X, "q")
            for b_ in (xv, xq):
                b_[:] = np.where(np.isfinite(b_), b_, np.nanmedian(b_))
            metodi = {"interpolazione": (xv, xq)}
            for n in ALBERI:
                metodi[f"lgb_{n}"] = (xv + modelli["v"].predict(X[colonne], num_iteration=n, num_threads=a.thread),
                                      xq + modelli["q"].predict(X[colonne], num_iteration=n, num_threads=a.thread))
            for metodo, (vh, qh) in metodi.items():
                pezzi.append(pd.DataFrame(dict(pannello=p, fold=k, g=X.g, t=X.t, l=X.l, mask_regime=X.mask_regime,
                                               metodo=metodo, v_hat=np.clip(vh, 3.0, None).astype(np.float32),
                                               q_hat=np.clip(qh * corsie, 50.0, None).astype(np.float32))))
        print(f"fold {k}: previsioni ({time.time() - t0:.0f} s)", flush=True)
    pd.concat(pezzi, ignore_index=True).to_parquet(CARTELLA / f"cv_previsioni_{a.allenamento}.parquet", index=False)


if __name__ == "__main__":
    main()
