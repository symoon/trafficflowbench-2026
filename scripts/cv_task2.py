"""Task 2: validazione a blocchi temporali sulle finestre ricostruite del train.

Per ogni finestra: storia dallo strato vero (come window_history, esente dalla maschera
del Task 1), riga a T dallo strato mascherato (come nei mesi valutati), etichette dalle
osservazioni vere. Onset: si prevede solo T+30. Ongoing: un modello su tutti i passi.
Le soglie si scelgono guardando la curva media sui tre fold, non un fold solo.

    uv run python scripts/cv_task2.py
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb.dati import RADICE_PROGETTO, tensori
from tfb.finestre import ORIZZONTE, STORIA, etichette
from tfb.task2_modello import caratteristiche, frequenza_coda, insieme, scarto_e1, togli_scarto

warnings.filterwarnings("ignore", category=RuntimeWarning)
CONFINI = (0, 92, 183, 273)
PANNELLI = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]
PARAMETRI = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 num_threads=8, verbose=-1, seed=0)
SOGLIE = np.round(np.arange(0.05, 0.96, 0.05), 2)
USCITA = RADICE_PROGETTO / "outputs" / "task2"


def blocco(g: int) -> int:
    return int(np.searchsorted(CONFINI, g, side="right") - 1)


def costruisci_righe(w: pd.DataFrame | None = None, e1: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Righe (finestra × tratto × passo) e descrizione delle finestre; w ha pannello, g, t, condition.

    Con e1 le velocità di storia e di T si riportano nel mondo del train (task2_modello.scarto_e1).
    """
    if w is None:
        w = pd.read_parquet(USCITA / "finestre_train.parquet")
    righe, finestre = [], []
    for p in PANNELLI:
        coda, ele = etichette(p)
        vero, masc = tensori(p, "train", "vero"), tensori(p, "train", "mascherato")
        okv = vero.eleggibile & np.isfinite(vero.speed)
        okm = masc.eleggibile & np.isfinite(masc.speed)
        freq = {k: frequenza_coda(coda, ele, np.r_[0:CONFINI[k], CONFINI[k + 1]:len(vero.date)]) for k in range(3)}
        if e1:
            from tfb.dati import tratti
            from tfb.task1 import livello_train
            vf = tratti(p).free_speed_kmh.to_numpy(np.float32)
            vs = np.where(okm, masc.speed, np.nan)
            mesi = pd.to_datetime(pd.Series(masc.date)).dt.to_period("M").to_numpy()
            livello = livello_train(p)
        for x in w[w.pannello == p].itertuples():
            g, t, k = x.g, x.t, blocco(x.g)
            st = slice(t - STORIA, t)
            sv, vT = np.where(okv[g, st], vero.speed[g, st], np.nan), np.where(okm[g, t], masc.speed[g, t], np.nan)
            if e1:
                d = scarto_e1(vs, mesi, g, t, livello)
                sv, vT = togli_scarto(sv, d, vf), togli_scarto(vT, d, vf)
            X = caratteristiche(
                p,
                sv, np.where(okv[g, st], vero.flow[g, st], np.nan),
                np.where(okv[g, st], vero.occ[g, st], np.nan),
                vT, np.where(okm[g, t], masc.flow[g, t], np.nan),
                np.where(okm[g, t], masc.occ[g, t], np.nan), t, freq[k])
            L = len(X)
            fid = len(finestre)
            fu = slice(t + 1, t + 1 + ORIZZONTE)
            finestre.append(dict(fid=fid, pannello=p, g=g, t=t, condition=x.condition, blocco=k,
                                 vero=coda[g, fu].copy(), ele=ele[g, fu].copy(), in_coda=X.in_coda.to_numpy() > 0.5))
            passi = [ORIZZONTE] if x.condition == "queue_onset" else range(1, ORIZZONTE + 1)
            for s in passi:
                Xs = X.copy()
                Xs["passo"] = np.float32(s)
                Xs["y"] = coda[g, t + s].astype(np.int8)
                Xs["ele"] = ele[g, t + s]
                Xs["fid"], Xs["l"], Xs["blocco"], Xs["condition"] = fid, np.arange(L), k, x.condition
                righe.append(Xs)
        print(f"{p}: {sum(1 for f in finestre if f['pannello'] == p)} finestre", flush=True)
    return pd.concat(righe, ignore_index=True), finestre


def iou_finestra(pred: np.ndarray, f: dict) -> float:
    e = f["ele"]
    u = np.logical_or(pred, f["vero"])[e].sum()
    return 1.0 if u == 0 else float(np.logical_and(pred, f["vero"])[e].sum() / u)


def main() -> None:
    t0 = time.time()
    d, finestre = costruisci_righe()
    meta = ["y", "ele", "fid", "l", "blocco", "condition"]
    colonne = [c for c in d.columns if c not in meta]
    print(f"{len(d):,} righe, {len(colonne)} caratteristiche ({time.time() - t0:.0f} s)", flush=True)
    prob = np.full(len(d), np.nan)
    for cond in ("queue_onset", "queue_ongoing"):
        for k in range(3):
            al = ((d.condition == cond) & (d.blocco != k) & d.ele).to_numpy()
            te = ((d.condition == cond) & (d.blocco == k)).to_numpy()
            m = lgb.train(PARAMETRI, lgb.Dataset(d.loc[al, colonne], d.y[al]), num_boost_round=400)
            prob[te] = m.predict(d.loc[te, colonne], num_threads=8)
            print(f"{cond} fold {k} ({time.time() - t0:.0f} s)", flush=True)
    d["p"] = prob
    per_fid = {fid: g for fid, g in d.groupby("fid")}
    risultati = []
    for f in finestre:
        g = per_fid[f["fid"]]
        L = f["vero"].shape[1]
        P = np.zeros((ORIZZONTE, L))
        P[g.passo.to_numpy().astype(int) - 1, g.l.to_numpy()] = g.p.to_numpy()
        base = dict(pannello=f["pannello"], condition=f["condition"], blocco=f["blocco"])
        risultati.append(dict(base, metodo="persistenza_T", iou=iou_finestra(np.broadcast_to(f["in_coda"], (ORIZZONTE, L)), f)))
        for s in SOGLIE:
            pred = insieme(P.ravel(), s, almeno_uno=True).reshape(P.shape)
            risultati.append(dict(base, metodo=f"lgb_{s:.2f}", iou=iou_finestra(pred, f)))
    r = pd.DataFrame(risultati)
    r.to_parquet(USCITA / "cv_finestre.parquet", index=False)
    # media per corridoio prima, come il valutatore (le famiglie qui pesano come i corridoi)
    curva = r.groupby(["condition", "metodo", "blocco", "pannello"]).iou.mean().groupby(["condition", "metodo", "blocco"]).mean().unstack("blocco")
    curva["media"] = curva.mean(axis=1)
    curva["min_fold"] = curva[[0, 1, 2]].min(axis=1)
    pd.set_option("display.width", 200)
    for cond in ("queue_onset", "queue_ongoing"):
        print(f"\n{cond}: IoU medio per fold\n", curva.loc[cond].round(3).to_string())


if __name__ == "__main__":
    main()
