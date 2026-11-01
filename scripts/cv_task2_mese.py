"""Task 2: le code dello stesso mese prima dell'origine T come caratteristiche.

I mesi valutati sono mondi diversi dal train: la quota di coda per tratto cambia di 2,5 punti
contro 1,3 tra i mesi del train (README, «Cosa sappiamo dei dati»), e la frequenza di coda del
modello viene tutta dal train. La regola del Task 2 ammette ogni dato con timestamp ≤ T, quindi
anche i giorni dello stesso mese prima della finestra. Per ogni finestra e tratto:
- mese_freq_ora, mese_n_ora: quota di celle osservate in coda e quante sono, nei giorni precedenti
  dello stesso mese, da T a T+12 (l'ora dell'orizzonte, come `frequenza` a T+6 ± 30 minuti);
- mese_scarto_ora: mese_freq_ora meno la frequenza del train nella stessa riga;
- mese_freq_tutto, mese_n_tutto: lo stesso su tutte le celle osservate del mese prima di T;
- ieri_coda: coda osservata sul tratto il giorno prima (nello stesso mese) da T a T+12; NaN se
  non c'è il giorno o non ci sono osservazioni.
Le osservazioni sono quelle che i mesi valutati rilasciano: lo strato mascherato del train (bersagli
del Task 1 nascosti) senza le celle oscurate da T'+1 a T'+18 attorno alle 10 finestre del mese
(cv_pseudomesi.finestre_mese), che negli split valutati non si vedono.

Probabilità con la ricetta di v8 (cv_task2_decoder.py): stesse righe, stessi fold, stessi modelli;
l'unica differenza sono le colonne nuove. Si confrontano con le probabilità di riferimento già
salvate da cv_task2_decoder.py probabilita, alle soglie dell'invio.

Regola dichiarata prima dei risultati: per condizione, le caratteristiche entrano solo se la media
sui fold migliora con l'intervallo bootstrap per giorno sopra lo zero sulle finestre ricostruite
(onset: attuali; ongoing: casuali dei tre semi), il fold peggiore non peggiora e sulle 80
ufficiali non si va nettamente nel verso opposto.

    uv run python scripts/cv_task2_mese.py righe
    uv run python scripts/cv_task2_mese.py probabilita [--thread 12]
    uv run python scripts/cv_task2_mese.py sintesi
"""
import argparse
import pickle
import time
import warnings
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from cv_task2 import PANNELLI, PARAMETRI, USCITA, iou_finestra
from cv_task2_decoder import ALLENAMENTO, INSIEMI, META, SOGLIE, almeno_uno, matrici, percorso
from tfb.dati import tensori, tratti
from tfb.finestre import ORIZZONTE, SOGLIA

warnings.filterwarnings("ignore", category=RuntimeWarning)
NUOVE = ["mese_freq_ora", "mese_n_ora", "mese_scarto_ora", "mese_freq_tutto", "mese_n_tutto", "ieri_coda"]
BANDA = 12  # da T a T+12
OSCURATI = 18  # da T'+1 a T'+18 attorno alle finestre valutate
PRIMI_GIORNI = 7


def calendario(p: str) -> tuple[np.ndarray, np.ndarray]:
    """Mese di ogni giorno del train e giorno nel mese (da 0)."""
    date = pd.to_datetime(pd.Series(tensori(p, "train", "vero").date))
    return date.dt.to_period("M").to_numpy(), date.dt.day.to_numpy() - 1


def osservazioni(p: str) -> tuple[np.ndarray, np.ndarray]:
    """(coda osservata, osservata) come negli split valutati: strato mascherato senza le celle oscurate."""
    from cv_pseudomesi import finestre_mese
    m = tensori(p, "train", "mascherato")
    vf = tratti(p).free_speed_kmh.to_numpy(np.float32)
    oss = m.eleggibile & np.isfinite(m.speed)
    w = finestre_mese(p, m.date)
    for g, t in zip(w.g, w.t):
        oss[g, t + 1:t + 1 + OSCURATI] = False
    return oss & (m.speed <= SOGLIA * vf), oss


def caratteristiche_mese(p: str, finestre: list[dict]) -> dict[int, np.ndarray]:
    """Per finestra (fid) la matrice (L, len(NUOVE)) delle caratteristiche del mese prima di T."""
    coda, oss = osservazioni(p)
    mesi, _ = calendario(p)
    D, T, L = coda.shape
    # totali per giorno, per sommare i giorni precedenti senza riscorrere le celle
    cg, ng = coda.sum(axis=1).astype(np.float32), oss.sum(axis=1).astype(np.float32)
    out = {}
    for f in finestre:
        g, t = f["g"], f["t"]
        g0 = int(np.flatnonzero(mesi == mesi[g])[0])
        b = slice(t, min(T, t + BANDA + 1))
        c_ora, n_ora = coda[g0:g, b].sum(axis=(0, 1)), oss[g0:g, b].sum(axis=(0, 1))
        c_tutto = cg[g0:g].sum(axis=0) + coda[g, :t + 1].sum(axis=0)
        n_tutto = ng[g0:g].sum(axis=0) + oss[g, :t + 1].sum(axis=0)
        ieri = np.full(L, np.nan, np.float32)
        if g > g0:
            vista = oss[g - 1, b].any(axis=0)
            ieri[vista] = coda[g - 1, b].any(axis=0)[vista]
        F = np.full((L, len(NUOVE)), np.nan, np.float32)
        F[:, 0] = np.where(n_ora > 0, c_ora / np.maximum(n_ora, 1), np.nan)
        F[:, 1] = n_ora
        F[:, 3] = np.where(n_tutto > 0, c_tutto / np.maximum(n_tutto, 1), np.nan)
        F[:, 4] = n_tutto
        F[:, 5] = ieri
        out[f["fid"]] = F
    return out


def righe() -> None:
    t0 = time.time()
    for n in INSIEMI:
        d = pd.read_parquet(percorso(n, "righe"), columns=["fid", "l", "frequenza"])
        fin = pickle.loads(percorso(n, "finestre").read_bytes())
        F = {}
        for p in PANNELLI:
            F.update(caratteristiche_mese(p, [f for f in fin if f["pannello"] == p]))
        L = np.array([F[f["fid"]].shape[0] for f in fin])
        inizio = np.r_[0, np.cumsum(L)[:-1]]
        tutto = np.concatenate([F[f["fid"]] for f in fin])
        pos = {f["fid"]: i for i, f in enumerate(fin)}
        X = tutto[inizio[d.fid.map(pos).to_numpy()] + d.l.to_numpy()]
        x = pd.DataFrame(X, columns=NUOVE)
        x["mese_scarto_ora"] = x.mese_freq_ora - d.frequenza.to_numpy(np.float32)
        x.to_parquet(USCITA / f"mese_righe_{n}.parquet", index=False)
        print(f"{n}: {len(x):,} righe, mese_freq_ora mancante {x.mese_freq_ora.isna().mean():.1%} "
              f"({time.time() - t0:.0f} s)", flush=True)


def probabilita(thread: int) -> None:
    """p_mese fuori campione dal fold esterno, con la ricetta e i fold di cv_task2_decoder.probabilita."""
    t0 = time.time()
    d = {n: pd.concat([pd.read_parquet(percorso(n, "righe")), pd.read_parquet(USCITA / f"mese_righe_{n}.parquet")], axis=1)
         for n in INSIEMI}
    colonne = [c for c in d["attuali"].columns if c not in META]
    par = dict(PARAMETRI, num_threads=thread)
    for n in INSIEMI:
        d[n]["p_mese"] = np.nan
    for cond, nomi in ALLENAMENTO.items():
        A = pd.concat([d[n][(d[n].condition == cond) & d[n].ele] for n in nomi], ignore_index=True)
        X, y, b = A[colonne].to_numpy(np.float32), A.y.to_numpy(), A.blocco.to_numpy()
        for k in range(3):
            mod = lgb.train(par, lgb.Dataset(X[b != k], y[b != k], feature_name=colonne), num_boost_round=400)
            for n in INSIEMI:
                te = ((d[n].condition == cond) & (d[n].blocco == k)).to_numpy()
                if te.any():
                    d[n].loc[te, "p_mese"] = mod.predict(d[n].loc[te, colonne].to_numpy(np.float32), num_threads=thread)
            if k == 0:
                imp = pd.Series(mod.feature_importance("gain"), index=colonne).sort_values(ascending=False)
                quota = imp / imp.sum()
                print(f"{cond}: quota del guadagno delle colonne nuove " +
                      ", ".join(f"{c} {quota[c]:.3f} ({list(imp.index).index(c) + 1}°)" for c in NUOVE), flush=True)
            print(f"{cond} fold {k} ({time.time() - t0:.0f} s)", flush=True)
    for n in INSIEMI:
        d[n][["fid", "condition", "blocco", "passo", "l", "p_mese"]].to_parquet(USCITA / f"mese_probabilita_{n}.parquet", index=False)


def sintesi() -> None:
    confronta("mese_probabilita", "p_mese", "mese_finestre")


def confronta(prefisso: str, colonna: str, uscita: str) -> None:
    """IoU alle soglie dell'invio: probabilità di riferimento (cv_task2_decoder) contro `colonna` di {prefisso}_{insieme}.parquet."""
    ris = []
    for n in INSIEMI:
        rif = pd.read_parquet(percorso(n, "probabilita"))
        nuovo = pd.read_parquet(USCITA / f"{prefisso}_{n}.parquet")
        assert (rif.fid.to_numpy() == nuovo.fid.to_numpy()).all() and (rif.l.to_numpy() == nuovo.l.to_numpy()).all()
        rif["p_nuovo"] = nuovo[colonna].to_numpy()
        fin = pickle.loads(percorso(n, "finestre").read_bytes())
        for p in PANNELLI:
            _, giorno = calendario(p)
            for f in fin:
                if f["pannello"] == p:
                    f["giorno_mese"] = int(giorno[f["g"]])
        for cond in SOGLIE:
            x = rif[rif.condition == cond]
            if x.empty:
                continue
            for col, metodo in (("p", "riferimento"), ("p_nuovo", "nuovo")):
                for fid, P in matrici(x, fin, col).items():
                    f = fin[fid]
                    cand = np.zeros(P.shape, bool)
                    if cond == "queue_onset":
                        cand[ORIZZONTE - 1] = True
                    else:
                        cand[:] = True
                    s = almeno_uno((P >= SOGLIE[cond]) & cand, np.where(cand, P, -1))
                    ris.append(dict(insieme=n, condition=cond, fold=f["blocco"], pannello=f["pannello"], g=f["g"], t=f["t"],
                                    giorno_mese=f["giorno_mese"], metodo=metodo, iou=iou_finestra(s, f)))
    r = pd.DataFrame(ris)
    r.to_parquet(USCITA / f"{uscita}.parquet", index=False)
    r["valutazione"] = np.where(r.insieme.str.startswith("casuali"), "casuali", r.insieme)
    pd.set_option("display.width", 200)
    for cond, valutazioni in (("queue_onset", ["attuali", "ufficiali"]), ("queue_ongoing", ["casuali", "attuali", "ufficiali"])):
        print(f"\n{cond}: IoU medio per corridoio, poi sui corridoi")
        for val in valutazioni:
            for nome, y in ((val, r[(r.condition == cond) & (r.valutazione == val)]),
                            (f"{val}, giorni 1–{PRIMI_GIORNI}", r[(r.condition == cond) & (r.valutazione == val) & (r.giorno_mese < PRIMI_GIORNI)])):
                if y.empty:
                    continue
                t = y.groupby(["metodo", "fold"]).apply(lambda z: z.groupby("pannello").iou.mean().mean()).unstack("fold")
                t["media"] = t.mean(axis=1)
                t["min_fold"] = t[[c for c in t.columns if c != "media"]].min(axis=1)
                d = t.loc["nuovo"] - t.loc["riferimento"]
                print(f"  {nome} ({len(y) // 2} finestre): riferimento {t.loc['riferimento', 'media']:.4f}, nuovo {t.loc['nuovo', 'media']:.4f}, "
                      f"Δ per fold " + " ".join(f"{d[c]:+.4f}" for c in t.columns if c not in ("media", "min_fold")) +
                      f", Δ fold peggiore {d['min_fold']:+.4f}" + boot(y))


def boot(y: pd.DataFrame, n: int = 1000) -> str:
    """Δ nuovo − riferimento con bootstrap per corridoio·giorno, aggregato per corridoio e poi sui corridoi."""
    w = y.pivot_table(index=["pannello", "g", "t", "insieme"], columns="metodo", values="iou").reset_index()
    idx = {k: v.to_numpy() for k, v in w.groupby(["pannello", "g"]).groups.items()}
    chiavi = list(idx)
    rng = np.random.default_rng(zlib.crc32(b"mese"))
    camp = []
    for _ in range(n):
        righe = np.concatenate([idx[chiavi[i]] for i in rng.integers(0, len(chiavi), len(chiavi))])
        m = w.loc[righe].groupby("pannello")[["nuovo", "riferimento"]].mean()
        camp.append(float((m.nuovo - m.riferimento).mean()))
    return f", bootstrap {np.mean(camp):+.4f} [{np.percentile(camp, 2.5):+.4f}, {np.percentile(camp, 97.5):+.4f}]"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["righe", "probabilita", "sintesi"])
    ap.add_argument("--thread", type=int, default=12)
    a = ap.parse_args()
    {"righe": righe, "probabilita": lambda: probabilita(a.thread), "sintesi": sintesi}[a.fase]()
