"""Task 2: più finestre di allenamento, come le copie dei buchi nel Task 1.

Oggi (v8) l'onset si allena sulle 2.102 finestre attuali (prima origine ammessa, 360 minuti tra
una finestra e l'altra) e l'ongoing sulle attuali più le casuali del seme 0. Qui:
- onset: attuali più tutte le origini con la storia senza coda e una coda in un passo qualunque
  dell'orizzonte, senza distanza minima (finestre.candidati_onset con solo_a_30=False, 16.447).
  Quelle con la coda a T+30 come le ufficiali sono solo 2.332: da sole non aggiungerebbero quasi
  nulla. Il modello si allena e si valuta sempre a T+30;
- ongoing: attuali più le casuali dei semi 0 … N−1 (--semi, 3 di default; i semi oltre il 2 si
  costruiscono con la fase `semi`).
Esito del 20/10: l'onset con le origini anticipate perde 0,026 ed è scartato (--senza-anticipo);
l'ongoing a 3 semi guadagna 0,010 sulle casuali.
Stessi parametri, fold e righe di valutazione di cv_task2_decoder.py; confronto con le sue
probabilità alle soglie dell'invio (cv_task2_mese.confronta). Regola di decisione dichiarata
prima: quella di cv_task2_mese.py.

    uv run python scripts/cv_task2_piu_finestre.py righe
    uv run python scripts/cv_task2_piu_finestre.py probabilita [--thread 12]
    uv run python scripts/cv_task2_piu_finestre.py sintesi
    uv run python scripts/cv_task2_piu_finestre.py semi --semi 6
    uv run python scripts/cv_task2_piu_finestre.py probabilita --semi 6 --senza-anticipo --nome piu_s6
    uv run python scripts/cv_task2_piu_finestre.py sintesi --nome piu_s6
"""
import argparse
import pickle
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from cv_task2 import PANNELLI, PARAMETRI, USCITA, costruisci_righe
from cv_task2_decoder import INSIEMI, META, percorso
from cv_task2_mese import confronta
from tfb.finestre import candidati_onset, ongoing_casuali

warnings.filterwarnings("ignore", category=RuntimeWarning)
SOGLIE_CURVA = (0.35, 0.40, 0.45, 0.50, 0.55)


def allenamento(semi: int, anticipo: bool) -> dict[str, list[str]]:
    return {"queue_onset": ["attuali"] + (["onset_anticipo"] if anticipo else []),
            "queue_ongoing": ["attuali"] + [f"casuali{s}" for s in range(semi)]}


def righe() -> None:
    t0 = time.time()
    w = pd.concat([candidati_onset(p, solo_a_30=False).assign(pannello=p, condition="queue_onset") for p in PANNELLI],
                  ignore_index=True)[["pannello", "g", "t", "condition"]]
    d, fin = costruisci_righe(w)
    d.to_parquet(percorso("onset_anticipo", "righe"), index=False)
    percorso("onset_anticipo", "finestre").write_bytes(pickle.dumps(fin))
    print(f"onset_anticipo: {len(fin)} finestre, {len(d):,} righe ({time.time() - t0:.0f} s)", flush=True)


def semi(n: int) -> None:
    """Righe delle ongoing casuali dei semi mancanti fino a n−1 (i semi 0–2 vengono da cv_task2_decoder.py righe)."""
    t0 = time.time()
    for s in range(n):
        if percorso(f"casuali{s}", "righe").exists():
            continue
        d, fin = costruisci_righe(pd.concat([ongoing_casuali(p, s) for p in PANNELLI], ignore_index=True))
        d.to_parquet(percorso(f"casuali{s}", "righe"), index=False)
        percorso(f"casuali{s}", "finestre").write_bytes(pickle.dumps(fin))
        print(f"casuali{s}: {len(fin)} finestre, {len(d):,} righe ({time.time() - t0:.0f} s)", flush=True)


def probabilita(thread: int, ALLENAMENTO: dict[str, list[str]], nome: str) -> None:
    t0 = time.time()
    nomi = sorted(set(INSIEMI) | {n for v in ALLENAMENTO.values() for n in v})
    d = {n: pd.read_parquet(percorso(n, "righe")) for n in nomi}
    colonne = [c for c in d["attuali"].columns if c not in META]
    par = dict(PARAMETRI, num_threads=thread)
    for n in INSIEMI:
        d[n]["p_piu"] = np.nan
    for cond, allena in ALLENAMENTO.items():
        A = pd.concat([d[n][(d[n].condition == cond) & d[n].ele] for n in allena], ignore_index=True)
        X, y, b = A[colonne].to_numpy(np.float32), A.y.to_numpy(), A.blocco.to_numpy()
        print(f"{cond}: {len(A):,} righe di allenamento", flush=True)
        for k in range(3):
            mod = lgb.train(par, lgb.Dataset(X[b != k], y[b != k]), num_boost_round=400)
            for n in INSIEMI:
                te = ((d[n].condition == cond) & (d[n].blocco == k)).to_numpy()
                if te.any():
                    d[n].loc[te, "p_piu"] = mod.predict(d[n].loc[te, colonne].to_numpy(np.float32), num_threads=thread)
            print(f"{cond} fold {k} ({time.time() - t0:.0f} s)", flush=True)
    for n in INSIEMI:
        d[n][["fid", "condition", "blocco", "passo", "l", "p_piu"]].to_parquet(USCITA / f"{nome}_probabilita_{n}.parquet", index=False)


def curva(nome: str) -> None:
    """IoU delle ongoing casuali per soglia, riferimento e nuovo, per fold: la soglia 0,45 deve restare sul plateau."""
    from cv_task2_decoder import almeno_uno, matrici
    from cv_task2 import iou_finestra
    r = []
    for n in ("casuali0", "casuali1", "casuali2"):
        rif = pd.read_parquet(percorso(n, "probabilita"))
        rif["p_piu"] = pd.read_parquet(USCITA / f"{nome}_probabilita_{n}.parquet").p_piu.to_numpy()
        rif = rif[rif.condition == "queue_ongoing"]
        fin = pickle.loads(percorso(n, "finestre").read_bytes())
        for col in ("p", "p_piu"):
            for fid, P in matrici(rif, fin, col).items():
                f = fin[fid]
                for s in SOGLIE_CURVA:
                    r.append(dict(col=col, soglia=s, fold=f["blocco"], pannello=f["pannello"], iou=iou_finestra(almeno_uno(P >= s, P), f)))
    t = pd.DataFrame(r).groupby(["col", "soglia", "fold", "pannello"]).iou.mean().groupby(["col", "soglia", "fold"]).mean().unstack("fold")
    t["media"], t["min_fold"] = t.mean(axis=1), t[[0, 1, 2]].min(axis=1)
    print("\nOngoing casuali: IoU per soglia (p = riferimento, p_piu = nuovo)\n" + t.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["righe", "semi", "probabilita", "sintesi"])
    ap.add_argument("--thread", type=int, default=12)
    ap.add_argument("--semi", type=int, default=3)
    ap.add_argument("--senza-anticipo", action="store_true")
    ap.add_argument("--nome", default="piu")
    a = ap.parse_args()
    if a.fase == "righe":
        righe()
    elif a.fase == "semi":
        semi(a.semi)
    elif a.fase == "probabilita":
        probabilita(a.thread, allenamento(a.semi, not a.senza_anticipo), a.nome)
    else:
        confronta(f"{a.nome}_probabilita", "p_piu", f"{a.nome}_finestre")
        curva(a.nome)
