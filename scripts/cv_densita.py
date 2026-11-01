"""Task 1/3: la densità dal rapporto di due modelli o da un modello suo, a blocchi temporali.

S_LWR guarda le variazioni di N = (q/v)·L nelle celle bersaglio: per una cella isolata
l'errore pesa 2·|N̂ - N|, in L1. Qui si confronta N̂ = q̂/v̂ (due modelli in L2) con un
modello della densità per corsia k = q/v, in L2 e in L1, da cui q̂ = k̂·v̂ oppure
v̂ = q̂/k̂. Stesse celle, caratteristiche e blocchi di cv_task1.py; per ogni fold si
allena sugli altri due blocchi e si valuta sui giorni completi del fold.

    uv run python scripts/cv_densita.py [--fold 0 1 2] [--alberi 800] [--frazione 1.0] [--thread 16]
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from cv_task1 import CARTELLA, META, valuta
from tfb.caratteristiche import base
from tfb.dati import pannelli, tensori

PARAMETRI = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 verbose=-1, seed=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--alberi", type=int, default=800)
    ap.add_argument("--frazione", type=float, default=1.0, help="quota delle righe di allenamento")
    ap.add_argument("--thread", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    d = pd.concat([pd.read_parquet(CARTELLA / "caratteristiche" / f"{p}.parquet") for p in pannelli()], ignore_index=True)
    colonne = [c for c in d.columns if c not in META]
    veri = {p: tensori(p, "train", "vero") for p in pannelli()}
    base_v, base_q = base(d, "v"), base(d, "q")
    for b_ in (base_v, base_q):
        med = d.assign(_b=b_).groupby(["pannello", "l"])._b.transform("median").to_numpy()
        b_[:] = np.where(np.isfinite(b_), b_, med)
    base_k = base_q / np.maximum(base_v, 1.0)
    y_k = d.y_q.to_numpy() / np.maximum(d.y_v.to_numpy(), 1.0)
    rng = np.random.default_rng(0)
    scelte = rng.random(len(d)) < a.frazione
    righe = []
    for k in a.fold:
        al = (d.blocco != k).to_numpy() & scelte
        te = ((d.blocco == k) & d.completo).to_numpy()
        dt = d[te].reset_index(drop=True)
        print(f"fold {k}: {al.sum():,} righe di allenamento, {te.sum():,} di valutazione ({time.time() - t0:.0f} s)", flush=True)
        pred = {}
        for nome, y, b_, obiettivo in (("v", d.y_v.to_numpy(), base_v, "regression"), ("q", d.y_q.to_numpy(), base_q, "regression"),
                                       ("k_l2", y_k, base_k, "regression"), ("k_l1", y_k, base_k, "l1")):
            res = y - b_
            ok = al & np.isfinite(res)
            m = lgb.train(dict(PARAMETRI, objective=obiettivo, num_threads=a.thread), lgb.Dataset(d.loc[ok, colonne], res[ok]),
                          num_boost_round=a.alberi)
            pred[nome] = b_[te] + m.predict(dt[colonne], num_threads=a.thread)
            print(f"fold {k} modello {nome} ({time.time() - t0:.0f} s)", flush=True)
        v, q = np.clip(pred["v"], 3.0, None), np.clip(pred["q"], 50.0 / 4, None)
        varianti = {
            "q̂/v̂ (come v6)": (v, q),
            "q = k̂_l2·v̂": (v, np.clip(pred["k_l2"], 0.1, None) * v),
            "q = k̂_l1·v̂": (v, np.clip(pred["k_l1"], 0.1, None) * v),
            "q = media(q̂, k̂_l1·v̂)": (v, 0.5 * (q + np.clip(pred["k_l1"], 0.1, None) * v)),
            "v = q̂/k̂_l1": (np.clip(q / np.clip(pred["k_l1"], 0.1, None), 3.0, None), q),
        }
        for nome, (vv, qq) in varianti.items():
            for r in valuta(dt, vv, qq, veri):
                righe.append(dict(r, metodo=nome, fold=k))
        tab = pd.DataFrame(righe)
        tab.to_csv(CARTELLA / "cv_densita.csv", index=False)
        sintesi(tab)


def sintesi(tab: pd.DataFrame) -> None:
    rif = "q̂/v̂ (come v6)"
    sint = tab.groupby("metodo", sort=False)[["S_state", "S_LWR", "rmse_v", "rmse_q"]].mean()
    sint["obiettivo"] = 0.35 * sint.S_state + 0.10 * sint.S_LWR
    per_fold = (0.35 * tab.S_state + 0.10 * tab.S_LWR).groupby([tab.metodo, tab.fold]).mean().unstack("fold")
    delta = per_fold.sub(per_fold.loc[rif], axis=1)
    sint["Δ obiettivo fold peggiore"] = delta.min(axis=1)
    base = tab[tab.metodo == rif].set_index(["pannello", "fold"]).S_LWR
    sint["corridoi·fold con S_LWR migliore"] = [
        f"{int((tab[tab.metodo == m].set_index(['pannello', 'fold']).S_LWR > base).sum())}/{len(base)}" for m in sint.index]
    pd.set_option("display.width", 220)
    print("\nobiettivo = 0,35·S_state + 0,10·S_LWR (la parte del totale che il Task 1 muove)\n", sint.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
