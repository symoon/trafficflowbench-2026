"""Task 2: decisione per finestra orientata all'IoU (E6) e decodifica temporale minima (E7).

REVISIONE_AVVERSARIALE_v2.md §10–11. Oggi la decisione è una soglia fissa sulle probabilità
per cella (onset 0,40 solo a T+30, ongoing 0,45 su tutto l'orizzonte, almeno una cella per
finestra). Qui la si confronta, sulle stesse probabilità fuori campione, con:
- topk_appross: le k celle più probabili, con k che massimizza Σ_{i∈S} p_i / (k + Σ_{i∉S} p_i).
  È un'approssimazione: il rapporto tra intersezione attesa e unione attesa non è l'IoU attesa;
- topk_mc: la stessa scelta di k con l'IoU attesa stimata per Monte Carlo da Bernoulli indipendenti;
- *_cal: le due precedenti su probabilità calibrate (isotonica) con le sole previsioni interne;
- viterbi (solo ongoing, E7): per tratto, la sequenza di stati sui 6 passi che massimizza
  Σ s_t (logit p_t − logit 0,45) − λ·(cambi di stato, partendo dallo stato a T); λ = 0 è la soglia.

Probabilità con la ricetta di v8: un modello per condizione, 400 alberi, PARAMETRI di cv_task2;
onset allenato sulle finestre attuali, ongoing su attuali più casuali (seme 0). Tre fold a
blocchi. Le previsioni interne vengono da modelli allenati su un solo blocco: per il fold
esterno k e una riga del blocco j ≠ k si usa il modello allenato sul terzo blocco. Calibrazione
e λ si stimano solo su queste previsioni interne delle righe di allenamento.

Valutazione sul blocco tenuto fuori: finestre attuali (onset e ongoing), ongoing casuali con i
semi 0, 1, 2 e le 80 finestre ufficiali del train (tutte nel blocco 0). IoU per finestra sulle
celle eleggibili, poi media per corridoio, poi sui corridoi.

Regola di scelta dichiarata prima di guardare i risultati: per condizione, il decoder con il
fold peggiore più alto sulle finestre ricostruite (onset: attuali; ongoing: casuali dei tre semi
insieme) diventa candidato solo se supera la soglia fissa anche nella media dei fold; poi lo si
misura sulle ufficiali. Altrimenti nessun cambio.

    uv run python scripts/cv_task2_decoder.py righe
    uv run python scripts/cv_task2_decoder.py probabilita [--thread 6]
    uv run python scripts/cv_task2_decoder.py decodifica
"""
import argparse
import pickle
import time
import warnings
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression

from cv_task2 import PANNELLI, PARAMETRI, USCITA, costruisci_righe, iou_finestra
from cv_task2_selezione import ufficiali
from tfb.finestre import ORIZZONTE, ongoing_casuali

warnings.filterwarnings("ignore", category=RuntimeWarning)
META = ["y", "ele", "fid", "l", "blocco", "condition"]
SOGLIE = {"queue_onset": 0.40, "queue_ongoing": 0.45}
INSIEMI = ["attuali", "casuali0", "casuali1", "casuali2", "ufficiali"]
ALLENAMENTO = {"queue_onset": ["attuali"], "queue_ongoing": ["attuali", "casuali0"]}
LAMBDA = (0.0, 0.25, 0.5, 1.0, 2.0)
MC = 400


def percorso(nome: str, tipo: str):
    return USCITA / f"decoder_{tipo}_{nome}.{'pkl' if tipo == 'finestre' else 'parquet'}"


def righe() -> None:
    t0 = time.time()
    attuali = pd.read_parquet(USCITA / "finestre_train.parquet")[["pannello", "g", "t", "condition"]]
    w = {"attuali": attuali, "ufficiali": ufficiali()}
    for s in range(3):
        w[f"casuali{s}"] = pd.concat([ongoing_casuali(p, s) for p in PANNELLI], ignore_index=True)
    for nome in INSIEMI:
        d, fin = costruisci_righe(w[nome])
        d.to_parquet(percorso(nome, "righe"), index=False)
        percorso(nome, "finestre").write_bytes(pickle.dumps(fin))
        print(f"{nome}: {len(fin)} finestre, {len(d):,} righe ({time.time() - t0:.0f} s)", flush=True)


def probabilita(thread: int) -> None:
    """p: fuori campione dal fold esterno; p_solo{m}: modello allenato sul solo blocco m (righe degli altri blocchi)."""
    t0 = time.time()
    d = {n: pd.read_parquet(percorso(n, "righe")) for n in INSIEMI}
    colonne = [c for c in d["attuali"].columns if c not in META]
    par = dict(PARAMETRI, num_threads=thread)
    for n in INSIEMI:
        d[n]["p"] = np.nan
        for m in range(3):
            d[n][f"p_solo{m}"] = np.nan
    for cond, nomi in ALLENAMENTO.items():
        A = pd.concat([d[n][(d[n].condition == cond) & d[n].ele] for n in nomi], ignore_index=True)
        X, y, b = A[colonne].to_numpy(np.float32), A.y.to_numpy(), A.blocco.to_numpy()
        for k in range(3):
            # fold esterno: allenato sugli altri due blocchi, previsto sul blocco k di ogni insieme
            mod = lgb.train(par, lgb.Dataset(X[b != k], y[b != k]), num_boost_round=400)
            for n in INSIEMI:
                te = ((d[n].condition == cond) & (d[n].blocco == k)).to_numpy()
                if te.any():
                    d[n].loc[te, "p"] = mod.predict(d[n].loc[te, colonne].to_numpy(np.float32), num_threads=thread)
            # modello interno: solo il blocco k, previsto sugli altri due blocchi delle righe di allenamento
            mod = lgb.train(par, lgb.Dataset(X[b == k], y[b == k]), num_boost_round=400)
            for n in nomi:
                te = ((d[n].condition == cond) & (d[n].blocco != k)).to_numpy()
                d[n].loc[te, f"p_solo{k}"] = mod.predict(d[n].loc[te, colonne].to_numpy(np.float32), num_threads=thread)
            print(f"{cond} fold {k} ({time.time() - t0:.0f} s)", flush=True)
    for n in INSIEMI:
        d[n][["fid", "condition", "blocco", "passo", "l", "ele", "y", "p"] + [f"p_solo{m}" for m in range(3)]].to_parquet(
            percorso(n, "probabilita"), index=False)


def calibratore(p: np.ndarray, y: np.ndarray, bins: int = 200):
    """Isotonica su quantili di p: restituisce una funzione p -> p calibrata."""
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    i = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    n = np.bincount(i, minlength=len(q) - 1).astype(float)
    ok = n > 0
    pm = np.bincount(i, weights=p, minlength=len(q) - 1)[ok] / n[ok]
    ym = np.bincount(i, weights=y, minlength=len(q) - 1)[ok] / n[ok]
    fit = isotonic_regression(ym, weights=n[ok]).x
    return lambda x: np.interp(x, pm, fit)


def topk(p: np.ndarray, mc: bool, rng) -> np.ndarray:
    """Insieme delle k celle più probabili che massimizza l'IoU attesa (approssimata o Monte Carlo); k ≥ 1."""
    o = np.argsort(-p, kind="stable")
    ps = p[o]
    k = np.arange(1, len(p) + 1)
    if mc:
        Y = rng.random((MC, len(p))) < ps[None, :]
        inter = np.cumsum(Y, axis=1)
        tot = Y.sum(axis=1, keepdims=True)
        e = (inter / (k[None, :] + tot - inter)).mean(axis=0)
    else:
        c = np.cumsum(ps)
        e = c / (k + (ps.sum() - c))
    s = np.zeros(len(p), bool)
    s[o[:int(np.argmax(e)) + 1]] = True
    return s


def viterbi(P: np.ndarray, s0: np.ndarray, soglia: float, lam: float) -> np.ndarray:
    """(6, L) binario: per tratto la sequenza che massimizza Σ s_t·z_t − λ·cambi, con z = logit p − logit soglia."""
    eps = 1e-6
    z = np.log(np.clip(P, eps, 1 - eps) / (1 - np.clip(P, eps, 1 - eps))) - np.log(soglia / (1 - soglia))
    T, L = P.shape
    v = np.stack([np.where(s0, -lam, 0.0), np.where(s0, 0.0, -lam)])  # punteggio per stato 0/1 prima del passo 1
    v = v + np.stack([np.zeros(L), z[0]])
    da = np.zeros((T, 2, L), np.int8)
    for t in range(1, T):
        nuovo = np.empty_like(v)
        for s in (0, 1):
            resta, cambia = v[s], v[1 - s] - lam
            da[t, s] = np.where(resta >= cambia, s, 1 - s)
            nuovo[s] = np.maximum(resta, cambia) + (z[t] if s else 0.0)
        v = nuovo
    out = np.zeros((T, L), bool)
    s = np.argmax(v, axis=0)
    for t in range(T - 1, -1, -1):
        out[t] = s == 1
        s = np.where(t > 0, da[t, s, np.arange(L)], s)
    return out


def almeno_uno(sel: np.ndarray, P: np.ndarray) -> np.ndarray:
    if not sel.any():
        sel = sel.copy()
        sel.flat[np.argmax(P)] = True
    return sel


def matrici(pr: pd.DataFrame, fin: list[dict], colonna: str) -> dict[int, np.ndarray]:
    """Per finestra la matrice (6, L) della colonna di probabilità indicata (0 dove manca)."""
    out = {}
    for fid, g in pr.groupby("fid"):
        L = fin[fid]["vero"].shape[1]
        P = np.zeros((ORIZZONTE, L))
        v = g[colonna].to_numpy()
        ok = np.isfinite(v)
        P[g.passo.to_numpy()[ok].astype(int) - 1, g.l.to_numpy()[ok]] = v[ok]
        out[fid] = P
    return out


def decodifica() -> None:
    t0 = time.time()
    pr = {n: pd.read_parquet(percorso(n, "probabilita")) for n in INSIEMI}
    fin = {n: pickle.loads(percorso(n, "finestre").read_bytes()) for n in INSIEMI}
    ris = []
    for cond, nomi in ALLENAMENTO.items():
        soglia = SOGLIE[cond]
        for k in range(3):
            # previsioni interne delle righe di allenamento fuori dal blocco k: riga del blocco j dal modello del terzo blocco
            interne = []
            for n in nomi:
                x = pr[n][(pr[n].condition == cond) & (pr[n].blocco != k)].copy()
                m = 3 - k - x.blocco.to_numpy()
                x["p_int"] = np.choose(m, [x.p_solo0.to_numpy(), x.p_solo1.to_numpy(), x.p_solo2.to_numpy()])
                interne.append((n, x))
            ok = pd.concat([x[x.ele] for _, x in interne])
            cal = calibratore(ok.p_int.to_numpy(), ok.y.to_numpy())
            lam = 0.0
            if cond == "queue_ongoing":
                # λ scelto sulle finestre di allenamento con le previsioni interne: IoU medio per corridoio, poi sui corridoi
                punti = {}
                mat = {n: matrici(x, fin[n], "p_int") for n, x in interne}
                for lam_ in LAMBDA:
                    r = []
                    for n, _ in interne:
                        for fid, P in mat[n].items():
                            f = fin[n][fid]
                            sel = almeno_uno(viterbi(P, f["in_coda"], soglia, lam_), P)
                            r.append((f["pannello"], iou_finestra(sel, f)))
                    punti[lam_] = pd.DataFrame(r, columns=["pannello", "iou"]).groupby("pannello").iou.mean().mean()
                lam = max(punti, key=punti.get)
                print(f"{cond} fold {k}: λ interno " + ", ".join(f"{a:g}: {b:.4f}" for a, b in punti.items()) + f" -> {lam:g}", flush=True)
            for n in INSIEMI:
                x = pr[n][(pr[n].condition == cond) & (pr[n].blocco == k)]
                if x.empty:
                    continue
                Pm = matrici(x, fin[n], "p")
                for fid, P in Pm.items():
                    f = fin[n][fid]
                    rng = np.random.default_rng([k, fid, zlib.crc32(n.encode())])
                    Pc = cal(P)
                    if cond == "queue_onset":  # solo T+30
                        cand = np.zeros(P.shape, bool); cand[ORIZZONTE - 1] = True
                    else:
                        cand = np.ones(P.shape, bool)
                    dec = {"soglia": almeno_uno((P >= soglia) & cand, np.where(cand, P, -1))}
                    for nome, Q, mc in (("topk_appross", P, False), ("topk_mc", P, True),
                                        ("topk_appross_cal", Pc, False), ("topk_mc_cal", Pc, True)):
                        s = np.zeros(P.shape, bool)
                        s[cand] = topk(Q[cand], mc, rng)
                        dec[nome] = s
                    if cond == "queue_ongoing":
                        dec["viterbi"] = almeno_uno(viterbi(P, f["in_coda"], soglia, lam), P)
                    base = dict(insieme=n, condition=cond, fold=k, pannello=f["pannello"], g=f["g"], t=f["t"], lam=lam)
                    for nome, s in dec.items():
                        ris.append(dict(base, decoder=nome, iou=iou_finestra(s, f), celle=int(s.sum())))
            print(f"{cond} fold {k} ({time.time() - t0:.0f} s)", flush=True)
    r = pd.DataFrame(ris)
    r.to_parquet(USCITA / "decoder_finestre.parquet", index=False)
    sintesi(r)


def media_corridoi(x: pd.DataFrame) -> float:
    return float(x.groupby("pannello").iou.mean().mean())


def sintesi(r: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    r = r.assign(valutazione=np.where(r.insieme.str.startswith("casuali"), "casuali", r.insieme))
    for cond, valutazioni in (("queue_onset", ["attuali", "ufficiali"]), ("queue_ongoing", ["casuali", "attuali", "ufficiali"])):
        x = r[r.condition == cond]
        print(f"\n{cond}: IoU medio per corridoio, poi sui corridoi")
        for val in valutazioni:
            y = x[x.valutazione == val]
            t = y.groupby(["decoder", "fold"]).apply(media_corridoi).unstack("fold")
            t["media"] = t.mean(axis=1)
            t["min_fold"] = t[[c for c in t.columns if c != "media"]].min(axis=1)
            # corridoi·fold migliorati rispetto alla soglia
            pf = y.groupby(["decoder", "fold", "pannello"]).iou.mean().unstack("decoder")
            t["meglio"] = [f"{int((pf[dname] > pf['soglia'] + 1e-12).sum())}/{int((pf[dname] != pf['soglia']).sum())} diversi"
                           for dname in t.index]
            t["finestre"] = y.groupby("decoder").size()
            print(f"  valutazione {val}\n" + t.round(4).to_string())
            boot(y)


def boot(y: pd.DataFrame, n: int = 1000) -> None:
    """Differenza dalla soglia con bootstrap per giorno (corridoio·giorno), aggregata per corridoio poi sui corridoi."""
    w = y.pivot_table(index=["pannello", "g", "t", "insieme"], columns="decoder", values="iou").reset_index()
    giorni = w[["pannello", "g"]].drop_duplicates().reset_index(drop=True)
    idx = {k: v.to_numpy() for k, v in w.groupby(["pannello", "g"]).groups.items()}
    rng = np.random.default_rng(0)
    altri = [c for c in w.columns if c not in ("pannello", "g", "t", "insieme", "soglia")]
    camp = {c: [] for c in altri}
    chiavi = list(zip(giorni.pannello, giorni.g))
    for _ in range(n):
        scelti = rng.integers(0, len(chiavi), len(chiavi))
        righe = np.concatenate([idx[chiavi[i]] for i in scelti])
        m = w.loc[righe].groupby("pannello")[altri + ["soglia"]].mean()
        for c in altri:
            camp[c].append(float((m[c] - m.soglia).mean()))
    print("    bootstrap per giorno, Δ dalla soglia: " + "; ".join(
        f"{c} {np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]" for c, v in camp.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["righe", "probabilita", "decodifica", "sintesi"])
    ap.add_argument("--thread", type=int, default=6)
    a = ap.parse_args()
    {"righe": righe, "probabilita": lambda: probabilita(a.thread), "decodifica": decodifica,
     "sintesi": lambda: sintesi(pd.read_parquet(USCITA / "decoder_finestre.parquet"))}[a.fase]()
