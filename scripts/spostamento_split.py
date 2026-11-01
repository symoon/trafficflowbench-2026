"""Validazione avversariale: i mesi valutati sono distinguibili dal train più di quanto lo siano due mesi del train?

Un classificatore LightGBM cerca di riconoscere da quale mese viene una cella osservata dello
strato mascherato (velocità, flusso per corsia, occupazione, istante, giorno della settimana,
tratto e stato dei vicini). Si confronta l'AUC di coppie di mesi consecutivi del train (la
variazione naturale tra un mese e l'altro) con febbraio contro validation e validation contro
private (mesi rigenerati l'11/09 con semi nuovi). La stima dell'AUC è fuori campione con 10 fold
per giorni, stratificati per mese: con due sole metà ogni proprietà del giorno (livello di domanda,
giorno della settimana) finisce in modo complementare tra allenamento e prova e spinge l'AUC sotto
0,5 (prima versione, 19/10). Per lo stesso motivo il giorno della settimana non è tra le caratteristiche.

    uv run python scripts/spostamento_split.py [--thread 16]
"""
import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from tfb.dati import pannelli, tensori, tratti

warnings.filterwarnings("ignore", category=RuntimeWarning)
CELLE_PER_MESE = 60_000


def celle(p: str, split: str, giorni: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    t = tensori(p, split)
    tr = tratti(p)
    corsie = tr.lanes.to_numpy(np.float32)
    vf = tr.free_speed_kmh.to_numpy(np.float32)
    ok = t.eleggibile & np.isfinite(t.speed) & np.isfinite(t.flow)
    ok[~np.isin(np.arange(len(t.date)), giorni)] = False
    g, ti, l = np.nonzero(ok)
    scelte = rng.choice(len(g), size=min(CELLE_PER_MESE, len(g)), replace=False)
    g, ti, l = g[scelte], ti[scelte], l[scelte]
    L = len(tr)
    monte, valle = np.clip(l - 1, 0, L - 1), np.clip(l + 1, 0, L - 1)
    return pd.DataFrame({
        "v": t.speed[g, ti, l], "q": t.flow[g, ti, l] / corsie[l], "o": t.occ[g, ti, l], "r": t.speed[g, ti, l] / vf[l],
        "v_monte": t.speed[g, ti, monte], "v_valle": t.speed[g, ti, valle],
        "q_monte": t.flow[g, ti, monte] / corsie[monte], "q_valle": t.flow[g, ti, valle] / corsie[valle],
        "istante": ti.astype(np.float32), "l": l.astype(np.float32),
        "giorno": g,
    })


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC come statistica di Mann–Whitney sui ranghi."""
    r = rankdata(p)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def auc(a: pd.DataFrame, b: pd.DataFrame, thread: int) -> float:
    """AUC fuori campione, 10 fold per giorni stratificati per mese."""
    x = pd.concat([a.assign(y=0), b.assign(y=1)], ignore_index=True)
    rng = np.random.default_rng(0)
    fold = np.zeros(len(x), int)
    for y in (0, 1):
        giorni = np.unique(x.giorno[x.y == y])
        assegna = dict(zip(rng.permutation(giorni), np.arange(len(giorni)) % 10))
        m = (x.y == y).to_numpy()
        fold[m] = x.giorno[m].map(assegna).to_numpy()
    col = [c for c in x.columns if c not in ("y", "giorno")]
    par = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
               num_threads=thread, verbose=-1, seed=0)
    p = np.zeros(len(x))
    for k in range(10):
        al, te = fold != k, fold == k
        m = lgb.train(par, lgb.Dataset(x.loc[al, col], x.y[al]), num_boost_round=200)
        p[te] = m.predict(x.loc[te, col], num_threads=thread)
    return roc_auc(x.y.to_numpy(), p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread", type=int, default=16)
    a = ap.parse_args()
    righe = []
    for p in pannelli():
        rng = np.random.default_rng(0)
        date = pd.to_datetime(pd.Series(tensori(p, "train").date))
        mesi = {m: np.flatnonzero((date.dt.to_period("M") == m).to_numpy()) for m in date.dt.to_period("M").unique()}
        chiavi = sorted(mesi)
        tr = {m: celle(p, "train", mesi[m], rng) for m in chiavi}
        val = celle(p, "validation", np.arange(len(tensori(p, "validation").date)), rng)
        pri = celle(p, "private", np.arange(len(tensori(p, "private").date)), rng)
        for m1, m2 in zip(chiavi[:-1], chiavi[1:]):
            righe.append(dict(pannello=p, coppia="train consecutivi", da=str(m1), a=str(m2), auc=auc(tr[m1], tr[m2], a.thread)))
        righe.append(dict(pannello=p, coppia="fine train → validation", da=str(chiavi[-1]), a="validation", auc=auc(tr[chiavi[-1]], val, a.thread)))
        righe.append(dict(pannello=p, coppia="validation → private", da="validation", a="private", auc=auc(val, pri, a.thread)))
        print(p, flush=True)
    r = pd.DataFrame(righe)
    r.to_csv("outputs/spostamento_split.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nAUC per corridoio (0,5 = indistinguibili)")
    print(r.pivot_table(index="pannello", columns="coppia", values="auc", aggfunc=["mean", "max"]).round(3).to_string())
    print("\nmedia sui corridoi:", r.groupby("coppia").auc.mean().round(3).to_dict())
    print("coppie del train con AUC più alto:", r[r.coppia == "train consecutivi"].sort_values("auc").tail(5)[["pannello", "da", "a", "auc"]].round(3).to_dict("records"))


if __name__ == "__main__":
    main()
