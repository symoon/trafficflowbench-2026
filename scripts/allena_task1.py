"""Task 1: allena LightGBM sui residui dell'interpolazione e lo valuta su giorni tenuti fuori.

Un solo modello per tutti i corridoi (flusso per corsia). Gli ultimi GIORNI_FUORI giorni
del train di ogni corridoio non entrano nell'allenamento e danno S_state e la stima
di S_LWR, con gli stessi conti dei valutatori.

    uv run python scripts/allena_task1.py [--campione 300000] [--pannelli D12_I5_N ...]
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb.caratteristiche import base, calcola
from tfb.dati import RADICE_PROGETTO, bersagli_task1, pannelli, tensori, tratti
from tfb.metrica import s_lwr_stima, s_state

FRAZIONE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
GIORNI_FUORI = 40
PARAMETRI = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 num_threads=16, verbose=-1, seed=0)


def prepara(p: str, campione: int, rng: np.random.Generator):
    vero, masc = tensori(p, "train", "vero"), tensori(p, "train", "mascherato")
    b = bersagli_task1(p, "train", masc)
    fuori = b.g >= len(masc.date) - GIORNI_FUORI
    dentro = np.flatnonzero(~fuori.to_numpy())
    scelti = np.sort(rng.choice(dentro, size=min(campione, len(dentro)), replace=False))
    celle = pd.concat([b.iloc[scelti], b[fuori]], ignore_index=True)
    X = calcola(masc, celle, FRAZIONE)
    corsie = tratti(p).lanes.to_numpy(np.float32)
    g, t, l = celle.g.to_numpy(), celle.t.to_numpy(), celle.l.to_numpy()
    y_v = vero.speed[g, t, l]
    y_q = vero.flow[g, t, l] / corsie[l]
    n_al = len(scelti)
    return X, y_v, y_q, n_al, celle.iloc[n_al:].reset_index(drop=True), vero, corsie


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campione", type=int, default=300_000)
    ap.add_argument("--pannelli", nargs="*")
    ap.add_argument("--round", type=int, default=600)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    elenco = a.pannelli or pannelli()
    t0 = time.time()
    Xa, va, qa, valuta = [], [], [], {}
    for p in elenco:
        X, y_v, y_q, n_al, celle_fuori, vero, corsie = prepara(p, a.campione, rng)
        Xa.append(X.iloc[:n_al]); va.append(y_v[:n_al]); qa.append(y_q[:n_al])
        valuta[p] = (X.iloc[n_al:].reset_index(drop=True), celle_fuori, vero, corsie)
        print(f"{p}: {n_al:,} celle di allenamento, {len(celle_fuori):,} fuori ({time.time() - t0:.0f} s)", flush=True)
    X = pd.concat(Xa, ignore_index=True)
    modelli = {}
    for nome, y in (("v", np.concatenate(va)), ("q", np.concatenate(qa))):
        r = y - base(X, nome)
        ok = np.isfinite(r)
        modelli[nome] = lgb.train(PARAMETRI, lgb.Dataset(X[ok], r[ok]), num_boost_round=a.round)
        print(f"modello {nome}: {ok.sum():,} righe ({time.time() - t0:.0f} s)", flush=True)

    risultati = {}
    for p, (Xf, celle, vero, corsie) in valuta.items():
        v_b, q_b = base(Xf, "v"), base(Xf, "q")
        v_hat = v_b + modelli["v"].predict(Xf, num_threads=16)
        q_hat = (q_b + modelli["q"].predict(Xf, num_threads=16)) * corsie[celle.l.to_numpy()]
        # dove non c'è né tempo né spazio (raro) resta NaN: si ripiega sull'interpolazione grezza
        v_hat = np.where(np.isfinite(v_hat), v_hat, np.nanmedian(v_b))
        q_hat = np.where(np.isfinite(q_hat), q_hat, np.nanmedian(q_b) * corsie[celle.l.to_numpy()])
        v_i = np.where(np.isfinite(v_b), v_b, np.nanmedian(v_b))
        q_i = np.where(np.isfinite(q_b), q_b, np.nanmedian(q_b)) * corsie[celle.l.to_numpy()]
        s_m, s_i = s_state(celle, vero, v_hat, q_hat), s_state(celle, vero, v_i, q_i)
        w_m, w_i = s_lwr_stima(celle, vero, v_hat, q_hat), s_lwr_stima(celle, vero, v_i, q_i)
        risultati[p] = dict(S_state=s_m["S_state"], S_state_interp=s_i["S_state"],
                            S_LWR=w_m["S_LWR"], S_LWR_interp=w_i["S_LWR"],
                            **{f"{r}_{k}": s_m[r][k] for r in ("R1", "R2", "R3") for k in ("rmse_v", "rmse_q")})
        print(f"{p}: S_state {s_m['S_state']:.4f} (interp {s_i['S_state']:.4f})  S_LWR~ {w_m['S_LWR']:.3f} (interp {w_i['S_LWR']:.3f})", flush=True)
    tab = pd.DataFrame(risultati).T
    tab.loc["media"] = tab.mean()
    print(tab.round(4).to_string())
    out = RADICE_PROGETTO / "outputs" / "task1"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "valutazione_lgb.csv")
    for nome, m in modelli.items():
        m.save_model(str(out / f"lgb_{nome}.txt"))
    imp = pd.DataFrame({n: m.feature_importance("gain") for n, m in modelli.items()}, index=X.columns)
    print((imp / imp.sum()).sort_values("v", ascending=False).round(3).head(25).to_string())
    (out / "parametri.json").write_text(json.dumps(dict(PARAMETRI, round=a.round, campione=a.campione), indent=1))


if __name__ == "__main__":
    main()
