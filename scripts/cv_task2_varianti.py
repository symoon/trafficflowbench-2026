"""Task 2: quanto dipende la stima locale dalla frequenza storica e dalla direzione del tempo.

Stesse finestre, caratteristiche e parametri di cv_task2.py; cambia solo come si calcola la
frequenza storica di coda (le colonne `frequenza` e `frequenza_T`) e su quali blocchi si allena.

- blocchi: come cv_task2.py; ogni finestra usa tutti i giorni fuori dal proprio blocco.
  Le righe di allenamento vedono così anche il blocco valutato (revisione del 19/10, punto 1).
- annidata: come blocchi, ma per le righe di allenamento si toglie anche il blocco valutato.
- senza: nessuna frequenza storica.
- progressiva: si allena solo sui blocchi precedenti a quello valutato (0 → 1, 0–1 → 2);
  le righe di allenamento usano i giorni prima della propria finestra, quelle valutate i giorni
  prima del blocco, come in produzione (train → validation e private).

Nella progressiva la soglia si sceglie sul passo 0 → 1 e si misura una volta sola sul passo 0–1 → 2.

    uv run python scripts/cv_task2_varianti.py [--thread 16]
"""
import argparse
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d

from cv_task2 import CONFINI, PANNELLI, PARAMETRI, SOGLIE, USCITA, costruisci_righe, iou_finestra
from tfb.dati import ISTANTI
from tfb.finestre import ORIZZONTE, etichette
from tfb.task2_modello import insieme

warnings.filterwarnings("ignore", category=RuntimeWarning)
SCELTE = {"queue_onset": 0.40, "queue_ongoing": 0.45}  # soglie dell'invio, scelte con la CV a blocchi


def cumulate(p: str) -> tuple[np.ndarray, np.ndarray]:
    """Conteggi di coda e di celle eleggibili, lisciati su ±15 minuti, cumulati sui giorni: (D+1, 288, L)."""
    coda, ele = etichette(p)
    k = np.ones(7, np.float32)
    c = convolve1d(coda.astype(np.float32), k, axis=1, mode="constant")
    n = convolve1d(ele.astype(np.float32), k, axis=1, mode="constant")
    z = np.zeros((1,) + c.shape[1:], np.float32)
    return np.concatenate([z, c.cumsum(0)]), np.concatenate([z, n.cumsum(0)])


def frequenza(C: np.ndarray, N: np.ndarray, intervalli: list[tuple[int, int]], righe: list[int]) -> np.ndarray:
    """Quota di giorni in coda sugli intervalli [a, b) di giorni, per gli istanti richiesti: (len(righe), L)."""
    c = sum(C[b, righe] - C[a, righe] for a, b in intervalli)
    n = sum(N[b, righe] - N[a, righe] for a, b in intervalli)
    return c / np.maximum(n, 1.0)


def fuori(*blocchi: int) -> list[tuple[int, int]]:
    return [(CONFINI[j], CONFINI[j + 1]) for j in range(3) if j not in blocchi]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    d, finestre = costruisci_righe()
    meta = ["y", "ele", "fid", "l", "blocco", "condition"]
    colonne = [c for c in d.columns if c not in meta]
    print(f"{len(d):,} righe ({time.time() - t0:.0f} s)", flush=True)
    cum = {p: cumulate(p) for p in PANNELLI}
    # righe di d per finestra, nell'ordine dei tratti: le frequenze si scrivono lì
    pos = d.groupby("fid").indices

    def con_frequenze(regola) -> pd.DataFrame:
        """Copia di d con frequenza e frequenza_T ricalcolate; regola(finestra) -> intervalli di giorni."""
        x = d.copy()
        fq, fT = x.frequenza.to_numpy().copy(), x.frequenza_T.to_numpy().copy()
        for f in finestre:
            C, N = cum[f["pannello"]]
            v = frequenza(C, N, regola(f), [min(f["t"] + ORIZZONTE, ISTANTI - 1), f["t"]])
            i = pos[f["fid"]]
            l = d.l.to_numpy()[i]
            fq[i], fT[i] = v[0, l], v[1, l]
        x["frequenza"], x["frequenza_T"] = fq, fT
        return x

    piani = []  # (variante, fold, dati di allenamento, dati di valutazione, maschera allenamento, maschera valutazione)
    blocchi = con_frequenze(lambda f: fuori(f["blocco"]))
    assert np.allclose(blocchi.frequenza.to_numpy(), d.frequenza.to_numpy(), atol=1e-5), "la variante blocchi deve rifare cv_task2"
    for k in range(3):
        piani.append(("blocchi", k, blocchi, blocchi, d.blocco != k, d.blocco == k))
        annidata = con_frequenze(lambda f, k=k: fuori(f["blocco"], k) if f["blocco"] != k else fuori(k))
        piani.append(("annidata", k, annidata, annidata, d.blocco != k, d.blocco == k))
        piani.append(("senza", k, d, d, d.blocco != k, d.blocco == k))
    for k in (1, 2):
        al = con_frequenze(lambda f: [(0, f["g"])])
        te = con_frequenze(lambda f, k=k: [(0, CONFINI[k])])
        piani.append(("progressiva", k, al, te, d.blocco < k, d.blocco == k))

    prob = {}
    for variante, k, dal, dte, mal, mte in piani:
        col = [c for c in colonne if not (variante == "senza" and c.startswith("frequenza"))]
        p = np.full(len(d), np.nan)
        for cond in ("queue_onset", "queue_ongoing"):
            al = ((d.condition == cond) & mal & d.ele).to_numpy()
            te = ((d.condition == cond) & mte).to_numpy()
            m = lgb.train(dict(PARAMETRI, num_threads=a.thread), lgb.Dataset(dal.loc[al, col], d.y[al]), num_boost_round=400)
            p[te] = m.predict(dte.loc[te, col], num_threads=a.thread)
        prob[(variante, k)] = p
        print(f"{variante} fold {k} ({time.time() - t0:.0f} s)", flush=True)

    righe = []
    for (variante, k), p in prob.items():
        dd = d.assign(p=p)
        per_fid = {fid: g for fid, g in dd[np.isfinite(p)].groupby("fid")}
        for f in finestre:
            if f["fid"] not in per_fid:
                continue
            g = per_fid[f["fid"]]
            P = np.zeros((ORIZZONTE, f["vero"].shape[1]))
            P[g.passo.to_numpy().astype(int) - 1, g.l.to_numpy()] = g.p.to_numpy()
            base = dict(variante=variante, fold=k, pannello=f["pannello"], g=f["g"], t=f["t"], condition=f["condition"])
            for s in SOGLIE:
                righe.append(dict(base, soglia=s, iou=iou_finestra(insieme(P.ravel(), s, almeno_uno=True).reshape(P.shape), f)))
    r = pd.DataFrame(righe)
    r.to_parquet(USCITA / "cv_varianti.parquet", index=False)

    # media per corridoio, poi sui corridoi; poi sui fold
    per_p = r.groupby(["variante", "condition", "soglia", "fold", "pannello"]).iou.mean().groupby(["variante", "condition", "soglia", "fold"]).mean()
    pd.set_option("display.width", 200)
    print("\nIoU alle soglie dell'invio (0,40 onset, 0,45 ongoing), per fold")
    sel = per_p.reset_index()
    sel = sel[sel.soglia.round(2) == sel.condition.map(SCELTE)]
    print(sel.pivot_table(index=["condition", "variante"], columns="fold", values="iou").round(3).to_string())
    print("\nSoglia migliore per variante (media sui fold) e valore")
    m = per_p.groupby(["variante", "condition", "soglia"]).mean().reset_index()
    print(m.loc[m.groupby(["variante", "condition"]).iou.idxmax()].round(3).to_string(index=False))
    pr = per_p.loc["progressiva"].reset_index()
    print("\nProgressiva: soglia scelta sul passo 0 → 1, misurata sul passo 0–1 → 2")
    for cond in SCELTE:
        x = pr[pr.condition == cond]
        s1 = x[x.fold == 1].sort_values("iou").iloc[-1].soglia
        v2 = x[(x.fold == 2) & (x.soglia == s1)].iou.iloc[0]
        print(f"  {cond}: soglia {s1:.2f} -> IoU blocco 2 = {v2:.3f} (con la soglia dell'invio {SCELTE[cond]:.2f}: "
              f"{x[(x.fold == 2) & (x.soglia.round(2) == SCELTE[cond])].iou.iloc[0]:.3f})")


if __name__ == "__main__":
    main()
