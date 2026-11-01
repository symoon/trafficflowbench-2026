"""Task 1: validazione a blocchi temporali (3 fold da tre mesi) su tutti i corridoi.

Legge le caratteristiche di prepara_task1.py, allena su due blocchi e valuta sui
giorni completi del terzo. Riporta S_state e la stima di S_LWR per corridoio e fold,
con media e dispersione, per l'interpolazione e per LightGBM a diversi numeri di alberi.

    uv run python scripts/cv_task1.py [--alberi 200 400 800] [--frazione 1.0] [--thread 16]
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb.caratteristiche import base
from tfb.dati import RADICE_PROGETTO, pannelli, tensori, tratti
from tfb.metrica import s_lwr_stima, s_state

CARTELLA = RADICE_PROGETTO / "outputs" / "task1"
META = ["y_v", "y_q", "g", "t", "l", "blocco", "completo", "mask_regime", "timestamp", "station_id", "link_id", "pannello"]
PARAMETRI = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 verbose=-1, seed=0)


def valuta(d: pd.DataFrame, v_hat: np.ndarray, q_corsia: np.ndarray, veri: dict) -> list[dict]:
    out = []
    for p, idx in d.groupby("pannello").indices.items():
        celle = d.iloc[idx]
        vero = veri[p]
        corsie = tratti(p).lanes.to_numpy(float)
        q_hat = q_corsia[idx] * corsie[celle.l.to_numpy()]
        s = s_state(celle, vero, v_hat[idx], q_hat)
        w = s_lwr_stima(celle, vero, v_hat[idx], q_hat)
        out.append(dict(pannello=p, S_state=s["S_state"], S_LWR=w["S_LWR"],
                        rmse_v=np.mean([s[r]["rmse_v"] for r in ("R1", "R2", "R3")]),
                        rmse_q=np.mean([s[r]["rmse_q"] for r in ("R1", "R2", "R3")])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alberi", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--frazione", type=float, default=1.0, help="quota delle righe di allenamento (curva di apprendimento)")
    ap.add_argument("--thread", type=int, default=16)
    a = ap.parse_args()
    PARAMETRI["num_threads"] = a.thread
    suffisso = "" if a.frazione == 1.0 else f"_frazione{a.frazione:g}"
    t0 = time.time()
    d = pd.concat([pd.read_parquet(CARTELLA / "caratteristiche" / f"{p}.parquet") for p in pannelli()], ignore_index=True)
    colonne = [c for c in d.columns if c not in META]
    veri = {p: tensori(p, "train", "vero") for p in pannelli()}
    print(f"{len(d):,} celle, {len(colonne)} caratteristiche ({time.time() - t0:.0f} s)", flush=True)
    base_v, base_q = base(d, "v"), base(d, "q")
    # dove manca anche lo spazio (tratto senza osservazioni nel giorno e nell'istante) si usa la mediana del tratto
    for b_, y in ((base_v, "y_v"), (base_q, "y_q")):
        med = d.assign(_b=b_).groupby(["pannello", "l"])._b.transform("median").to_numpy()
        b_[:] = np.where(np.isfinite(b_), b_, med)
    righe = []
    scelte = np.random.default_rng(1).random(len(d)) < a.frazione
    for k in range(3):
        al = (d.blocco != k).to_numpy() & scelte
        te = ((d.blocco == k) & d.completo).to_numpy()
        dt = d[te].reset_index(drop=True)
        for r in valuta(dt, base_v[te], base_q[te], veri):
            righe.append(dict(r, fold=k, metodo="interpolazione"))
        pred = {}
        for nome, y, b_ in (("v", d.y_v.to_numpy(), base_v), ("q", d.y_q.to_numpy(), base_q)):
            res = y - b_
            ok = al & np.isfinite(res)
            m = lgb.train(PARAMETRI, lgb.Dataset(d.loc[ok, colonne], res[ok]), num_boost_round=max(a.alberi))
            pred[nome] = {n: b_[te] + m.predict(dt[colonne], num_iteration=n, num_threads=a.thread) for n in a.alberi}
            print(f"fold {k} modello {nome} ({time.time() - t0:.0f} s)", flush=True)
        for n in a.alberi:
            for r in valuta(dt, pred["v"][n], pred["q"][n], veri):
                righe.append(dict(r, fold=k, metodo=f"lgb_{n}"))
        tab = pd.DataFrame(righe)
        print(tab[tab.fold == k].groupby("metodo")[["S_state", "S_LWR", "rmse_v", "rmse_q"]].mean().round(4).to_string(), flush=True)
    tab = pd.DataFrame(righe)
    tab.to_csv(CARTELLA / f"cv_blocchi{suffisso}.csv", index=False)
    per_fold = tab.groupby(["metodo", "fold"])[["S_state", "S_LWR"]].mean().unstack("fold")
    print("\nMedia sui corridoi per fold\n", per_fold.round(4).to_string())
    sint = tab.groupby("metodo").agg(S_state=("S_state", "mean"), S_state_dev_fold=("S_state", lambda s: 0.0),
                                     S_LWR=("S_LWR", "mean"))
    sint["S_state_dev_fold"] = tab.groupby(["metodo", "fold"]).S_state.mean().groupby("metodo").std()
    sint["S_state_min_corridoio"] = tab.groupby(["metodo", "pannello"]).S_state.mean().groupby("metodo").min()
    print("\nSintesi\n", sint.round(4).to_string())
    per_p = tab.pivot_table(index="pannello", columns="metodo", values="S_state", aggfunc="mean")
    print("\nS_state per corridoio (media dei fold)\n", per_p.round(4).to_string())
    (CARTELLA / f"cv_parametri{suffisso}.json").write_text(json.dumps(dict(PARAMETRI, alberi=a.alberi, frazione=a.frazione), indent=1))


if __name__ == "__main__":
    main()
