"""Task 2: allenare le ongoing su finestre scelte come quelle ufficiali.

Le ongoing ufficiali del train cadono in posizioni sparse tra le origini ammesse; la
selezione attuale prende sempre la prima origine ammessa. Qui si confronta l'allenamento su
selezione attuale, selezione casuale (finestre.ongoing_casuali) e unione delle due.

Banco di prova pulito: le 80 finestre ufficiali del train stanno tutte nel blocco 0, quindi
un modello allenato sui blocchi 1 e 2 le prevede fuori campione. Si valuta anche sulle
finestre attuali e casuali del blocco 0, e con tre fold sulle casuali.

    uv run python scripts/cv_task2_selezione.py [--thread 16]
"""
import argparse
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from cv_task2 import PANNELLI, PARAMETRI, SOGLIE, USCITA, costruisci_righe, iou_finestra
from tfb.dati import _indice_istante, tensori
from tfb.finestre import ORIZZONTE, ongoing_casuali
from tfb.task2 import finestre as finestre_ufficiali
from tfb.task2_modello import insieme

warnings.filterwarnings("ignore", category=RuntimeWarning)
META = ["y", "ele", "fid", "l", "blocco", "condition"]


def ufficiali() -> pd.DataFrame:
    righe = []
    for p in PANNELLI:
        w = finestre_ufficiali(p, "train")
        gi = {d: i for i, d in enumerate(tensori(p, "train", "vero").date)}
        righe.append(pd.DataFrame({"pannello": p, "g": w.date.map(gi).to_numpy(), "t": _indice_istante(w.forecast_origin),
                                   "condition": w.condition.to_numpy()}))
    return pd.concat(righe, ignore_index=True)


def valuta(d: pd.DataFrame, fin: list[dict], p: np.ndarray, nome: str, extra: dict) -> list[dict]:
    dd = d.assign(p=p)
    per_fid = {fid: g for fid, g in dd[np.isfinite(p)].groupby("fid")}
    out = []
    for f in fin:
        if f["fid"] not in per_fid:
            continue
        g = per_fid[f["fid"]]
        P = np.zeros((ORIZZONTE, f["vero"].shape[1]))
        P[g.passo.to_numpy().astype(int) - 1, g.l.to_numpy()] = g.p.to_numpy()
        base = dict(extra, valutazione=nome, pannello=f["pannello"], g=f["g"], t=f["t"], condition=f["condition"])
        out.append(dict(base, soglia=-1.0, iou=iou_finestra(np.broadcast_to(f["in_coda"], P.shape), f)))  # persistenza da T
        for s in SOGLIE:
            out.append(dict(base, soglia=s, iou=iou_finestra(insieme(P.ravel(), s, almeno_uno=True).reshape(P.shape), f)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    attuali = pd.read_parquet(USCITA / "finestre_train.parquet")
    casuali = pd.concat([ongoing_casuali(p) for p in PANNELLI], ignore_index=True)
    insiemi = {"attuali": attuali[attuali.condition == "queue_ongoing"][["pannello", "g", "t", "condition"]],
               "casuali": casuali, "ufficiali": ufficiali().query("condition == 'queue_ongoing'")}
    righe = {k: costruisci_righe(w) for k, w in insiemi.items()}
    for k, (d, fin) in righe.items():
        print(f"{k}: {len(fin)} finestre ongoing, {len(d):,} righe ({time.time() - t0:.0f} s)", flush=True)
    colonne = [c for c in righe["attuali"][0].columns if c not in META]
    par = dict(PARAMETRI, num_threads=a.thread)

    def allena(nomi: list[str], esclusi: int) -> lgb.Booster:
        X = pd.concat([righe[n][0][(righe[n][0].blocco != esclusi) & righe[n][0].ele] for n in nomi], ignore_index=True)
        return lgb.train(par, lgb.Dataset(X[colonne], X.y), num_boost_round=400)

    ris = []
    for nome, nomi in (("attuali", ["attuali"]), ("casuali", ["casuali"]), ("unione", ["attuali", "casuali"])):
        m = allena(nomi, 0)
        for val in ("ufficiali", "casuali", "attuali"):
            d, fin = righe[val]
            te = (d.blocco == 0).to_numpy()
            p = np.full(len(d), np.nan)
            p[te] = m.predict(d.loc[te, colonne], num_threads=a.thread)
            ris += valuta(d, [f for f in fin if f["blocco"] == 0], p, val, dict(allenamento=nome, fold=0))
        print(f"allenamento {nome} ({time.time() - t0:.0f} s)", flush=True)
    # tre fold sulle casuali, allenando su attuali o casuali: stima per finestre come le ufficiali
    for nome, nomi in (("attuali", ["attuali"]), ("casuali", ["casuali"])):
        for k in (1, 2):
            m = allena(nomi, k)
            d, fin = righe["casuali"]
            te = (d.blocco == k).to_numpy()
            p = np.full(len(d), np.nan)
            p[te] = m.predict(d.loc[te, colonne], num_threads=a.thread)
            ris += valuta(d, [f for f in fin if f["blocco"] == k], p, "casuali", dict(allenamento=nome, fold=k))
        print(f"fold 1-2 allenamento {nome} ({time.time() - t0:.0f} s)", flush=True)
    r = pd.DataFrame(ris)
    r.to_parquet(USCITA / "cv_selezione.parquet", index=False)
    media = r.groupby(["allenamento", "valutazione", "fold", "soglia", "pannello"]).iou.mean().groupby(["allenamento", "valutazione", "fold", "soglia"]).mean()
    pd.set_option("display.width", 200)
    f0 = media.xs(0, level="fold").unstack("soglia")
    print("\nFold 0 (blocchi 1-2 -> blocco 0), IoU medio per corridoio; soglia -1 = persistenza da T")
    print(f0[[-1.0, 0.3, 0.4, 0.45, 0.5, 0.6]].round(3).to_string())
    print("\nNumero di finestre valutate:", r[(r.soglia == 0.45) & (r.fold == 0)].groupby(["allenamento", "valutazione"]).size().to_dict())
    tre = media.xs("casuali", level="valutazione").unstack("soglia")[[-1.0, 0.4, 0.45, 0.5]]
    print("\nValutazione sulle casuali, per fold\n", tre.round(3).to_string())


if __name__ == "__main__":
    main()
