"""Task 1, E5: completamento a basso rango del mese come caratteristica (REVISIONE_AVVERSARIALE_v2, E5).

I dati sono un profilo per tratto più rumore bianco (2–3% sul flusso, circa 1,2 km/h sulla velocità):
il predicibile di una cella è il profilo del suo tratto, che il mese intero stima molto meglio di due
vicini rumorosi. Per ogni corridoio e mese la matrice (giorno × istante) × tratto delle osservazioni
eleggibili dello strato mascherato si completa a rango r (minimi quadrati alternati con cresta),
per la velocità e per il flusso per corsia. Le stime entrano come caratteristiche: lr_v, lr_q e la loro
differenza dall'interpolazione nel tempo. Il mese è valutato nel suo blocco, quindi i fold non si
contaminano (i mesi stanno tutti dentro un blocco).

Fasi:
- rango: sceglie r tenendo da parte il 10% delle celle osservate di due mesi per corridoio (nessun bersaglio);
- prepara: caratteristiche E5 per le righe di caratteristiche/ (prepara_task1.py), stesso ordine;
- cv: tre blocchi come cv_task1.py, ramo std con e senza E5, 800 alberi; S_state, S_LWR stimato e la parte
  del totale 0,35·S_state + 0,10·S_LWR, per corridoio e fold.

Regola dichiarata prima dei risultati: E5 passa ai rami dell'invio solo se la parte del totale sale in
almeno 25 corridoi·fold su 30 e di almeno +0,0005 in media.

    uv run python scripts/e5_basso_rango.py rango
    uv run python scripts/e5_basso_rango.py prepara --rango-v 4 --rango-q 4
    uv run python scripts/e5_basso_rango.py cv [--thread 16] [--mese]
    uv run python scripts/e5_basso_rango.py prepara_mondi
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from cv_task1 import CARTELLA, META, PARAMETRI, valuta
from tfb.caratteristiche import E5, base, completa_mese, mesi
from tfb.dati import pannelli, tensori, tratti
from tfb.task1 import evidenza

USCITA = CARTELLA / "caratteristiche_e5"
E5 = list(E5)
completa = completa_mese


def rango() -> None:
    rng = np.random.default_rng(5)
    righe = []
    for p in pannelli():
        t = tensori(p, "train", "mascherato")
        vs, qs = evidenza(t)
        corsie = tratti(p).lanes.to_numpy(np.float32)
        m = mesi(t)
        due = np.unique(m)[[1, 6]]  # due mesi, uno per stagione
        tenere = np.isfinite(vs) & (rng.random(vs.shape) < 0.10) & np.isin(m, due)[:, None, None]
        for r in (1, 2, 4, 8, 12):
            lv, lq = completa(t, r, r, tenere, soli=due)
            ev = tenere & np.isfinite(lv)
            righe.append(dict(pannello=p, rango=r, rmse_v=float(np.sqrt(np.nanmean((lv - vs)[ev] ** 2))),
                              rmse_q=float(np.sqrt(np.nanmean((lq - qs / corsie)[ev] ** 2)))))
        print(pd.DataFrame(righe)[lambda z: z.pannello == p].round(3).to_string(index=False), flush=True)
    tab = pd.DataFrame(righe).groupby("rango")[["rmse_v", "rmse_q"]].mean()
    print("\nmedia sui corridoi (celle osservate tenute da parte)\n" + tab.round(3).to_string())


def prepara(rango_v: int, rango_q: int) -> None:
    USCITA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for p in pannelli():
        d = pd.read_parquet(CARTELLA / "caratteristiche" / f"{p}.parquet", columns=["g", "t", "l", "v_it", "q_it"])
        lv, lq = completa(tensori(p, "train", "mascherato"), rango_v, rango_q)
        g, t, l = d.g.to_numpy(), d.t.to_numpy(), d.l.to_numpy()
        x = pd.DataFrame({"lr_v": lv[g, t, l], "lr_q": lq[g, t, l]})
        x["lr_v_meno_it"] = x.lr_v - d.v_it.to_numpy()
        x["lr_q_meno_it"] = x.lr_q - d.q_it.to_numpy()
        x.astype(np.float32).to_parquet(USCITA / f"{p}.parquet", index=False)
        print(f"{p}: {len(x):,} celle ({time.time() - t0:.0f} s)", flush=True)
    (USCITA / "parametri.json").write_text(json.dumps(dict(rango_v=rango_v, rango_q=rango_q, cresta=1.0, giri=15)))


def prepara_mondi() -> None:
    """E5 per le righe di caratteristiche_mondi/ (prepara_task1_mondi.py, copia 0): lo stesso mondo spostato di ogni blocco."""
    import zlib

    from cv_mondo import SD_MONDO, sposta
    t0 = time.time()
    for p in pannelli():
        d = pd.read_parquet(CARTELLA / "caratteristiche_mondi" / f"{p}.parquet", columns=["g", "t", "l", "blocco", "copia", "v_it", "q_it"])
        assert (d.copia == 0).all(), "solo la copia 0 dei mondi"
        masc, vero = tensori(p, "train"), tensori(p, "train", "vero")
        vf = tratti(p).free_speed_kmh.to_numpy(np.float32)
        rif = np.where(np.isfinite(vero.speed), vero.speed, np.nanmedian(vero.speed))
        m_mese = mesi(masc)
        x = np.full((len(d), 2), np.nan, np.float32)
        for k in range(3):
            rng = np.random.default_rng([0, k, zlib.crc32(p.encode())])  # come prepara_task1_mondi.py
            m = sposta(masc, vf, rng.normal(0.0, SD_MONDO, len(vf)).astype(np.float32), rif)
            r = (d.blocco == k).to_numpy()
            g, t, l = d.g.to_numpy()[r], d.t.to_numpy()[r], d.l.to_numpy()[r]
            lv, lq = completa_mese(m, soli=np.unique(m_mese[g]))
            x[r, 0], x[r, 1] = lv[g, t, l], lq[g, t, l]
        o = pd.DataFrame({"lr_v": x[:, 0], "lr_q": x[:, 1]})
        o["lr_v_meno_it"] = o.lr_v - d.v_it.to_numpy()
        o["lr_q_meno_it"] = o.lr_q - d.q_it.to_numpy()
        o.astype(np.float32).to_parquet(USCITA / f"{p}__mondi.parquet", index=False)
        print(f"{p} mondi: {len(o):,} celle ({time.time() - t0:.0f} s)", flush=True)


def cv(thread: int, mese: bool) -> None:
    """Con mese il riferimento è il ramo dell'invio: caratteristiche di base più il livello del mese (cv_mese.py)."""
    PARAMETRI["num_threads"] = thread
    t0 = time.time()
    d = pd.concat([pd.read_parquet(CARTELLA / "caratteristiche" / f"{p}.parquet") for p in pannelli()], ignore_index=True)
    e = pd.concat([pd.read_parquet(USCITA / f"{p}.parquet") for p in pannelli()], ignore_index=True)
    parti = [d, e]
    if mese:
        parti.append(pd.concat([pd.read_parquet(CARTELLA / "caratteristiche_mese" / f"{p}.parquet") for p in pannelli()], ignore_index=True))
    d = pd.concat(parti, axis=1)
    std = [c for c in d.columns if c not in META + E5]
    veri = {p: tensori(p, "train", "vero") for p in pannelli()}
    base_v, base_q = base(d, "v"), base(d, "q")
    for b_ in (base_v, base_q):
        med = d.assign(_b=b_).groupby(["pannello", "l"])._b.transform("median").to_numpy()
        b_[:] = np.where(np.isfinite(b_), b_, med)
    print(f"{len(d):,} celle ({time.time() - t0:.0f} s)", flush=True)
    righe = []
    for k in range(3):
        al = (d.blocco != k).to_numpy()
        te = ((d.blocco == k) & d.completo).to_numpy()
        dt = d[te].reset_index(drop=True)
        for nome_var, colonne in (("std", std), ("std+e5", std + E5)):  # con mese, «std» è il ramo mese
            pred = {}
            for nome, y, b_ in (("v", d.y_v.to_numpy(), base_v), ("q", d.y_q.to_numpy(), base_q)):
                res = y - b_
                ok = al & np.isfinite(res)
                m = lgb.train(PARAMETRI, lgb.Dataset(d.loc[ok, colonne], res[ok]), num_boost_round=800)
                pred[nome] = b_[te] + m.predict(dt[colonne], num_threads=thread)
            for r in valuta(dt, pred["v"], pred["q"], veri):
                righe.append(dict(r, fold=k, metodo=nome_var))
            print(f"fold {k} {nome_var} ({time.time() - t0:.0f} s)", flush=True)
    r = pd.DataFrame(righe)
    r["parte"] = 0.35 * r.S_state + 0.10 * r.S_LWR
    r.to_csv(CARTELLA / ("e5_cv_mese.csv" if mese else "e5_cv.csv"), index=False)
    w = r.pivot_table(index=["pannello", "fold"], columns="metodo", values=["parte", "S_state", "S_LWR", "rmse_v", "rmse_q"])
    d_ = {c: w[(c, "std+e5")] - w[(c, "std")] for c in ("parte", "S_state", "S_LWR", "rmse_v", "rmse_q")}
    print("\nmedia su corridoi e fold\n" + r.groupby("metodo")[["S_state", "S_LWR", "rmse_v", "rmse_q", "parte"]].mean().round(5).to_string())
    print("\nΔ std+e5 − std: " + ", ".join(f"{c} {v.mean():+.5f}" for c, v in d_.items()) +
          f"; parte meglio in {(d_['parte'] > 0).sum()} su {len(d_['parte'])} corridoi·fold, minimo {d_['parte'].min():+.5f}")
    print("\nper corridoio (Δ parte, media sui fold)\n" + d_["parte"].groupby(level=0).mean().round(5).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["rango", "prepara", "prepara_mondi", "cv"])
    ap.add_argument("--rango-v", type=int, default=4)
    ap.add_argument("--rango-q", type=int, default=4)
    ap.add_argument("--thread", type=int, default=16)
    ap.add_argument("--mese", action="store_true", help="cv: riferimento con il livello del mese (ramo dell'invio)")
    a = ap.parse_args()
    {"rango": rango, "prepara": lambda: prepara(a.rango_v, a.rango_q), "prepara_mondi": prepara_mondi,
     "cv": lambda: cv(a.thread, a.mese)}[a.fase]()
