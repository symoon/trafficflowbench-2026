"""Task 2: altri motori accanto a LightGBM, stacking e dove sbaglia il modello (28/10/2026).

Domande:
- XGBoost o CatBoost accanto a LightGBM, con la media delle probabilità? Nel Task 1 XGBoost
  sbagliava come LightGBM (20/10). CatBoost usa alberi simmetrici e un boosting ordinato, quindi può
  sbagliare in modo diverso. Configurazioni standard, non regolate.
- Stack: un secondo LightGBM sulle stesse caratteristiche più la probabilità del primo (stack). Poi
  lo stesso con il contesto della probabilità del primo nella finestra (stack_ctx): tratti vicini,
  massimo a monte e a valle, massimo, rango e somma nel passo, per le ongoing anche i passi vicini e
  la media sul tratto. Poi con le probabilità dei tre motori (stack_motori): una media appresa, che
  può dare pesi diversi ai motori in zone diverse.
- Dove sbaglia: IoU per fasce di corridoio, ora, dimensione della coda vera e frequenza storica dei
  tratti in coda; per l'onset anche gli errori per cella per fasce di frequenza e di velocità a T.

Primo livello: la ricetta dell'invio (v15). PARAMETRI di cv_task2, 400 alberi, onset sulle finestre
attuali a T+30, ongoing sulle attuali più le casuali dei semi 0–5. Tre fold a blocchi.

Stacking annidato. Per il fold esterno k, le righe di allenamento del blocco j ricevono la
probabilità del modello allenato sul solo terzo blocco m. Le righe del blocco k ricevono quella del
modello esterno, allenato su j e m. Così il secondo livello non vede mai una probabilità calcolata
sulle proprie righe, come all'invio, dove le righe del train avranno le previsioni fuori fold.

Regola dichiarata prima dei risultati: per condizione, un metodo diventa candidato solo se, alla
soglia dell'invio (onset 0,40, ongoing 0,45), batte LightGBM nella media dei fold e nel fold peggiore
sulle finestre ricostruite (onset: attuali; ongoing: casuali dei semi 0–2), con l'intervallo bootstrap
per corridoio·giorno sopra lo zero. Tra i candidati vince il più semplice (media < stack < stack_ctx
< stack_motori), se la differenza dal migliore sta dentro l'intervallo. Le finestre ufficiali del
train sono solo un controllo. La curva delle soglie serve a vedere se la soglia resta sul plateau.
Le fasce di `debolezze` sono diagnosi: un commutatore per fascia scelto su questi numeri andrebbe
validato su un fold che non li ha visti.

    uv run python scripts/cv_task2_stack.py primo --condizione onset
    uv run python scripts/cv_task2_stack.py secondo --condizione onset
    uv run python scripts/cv_task2_stack.py sintesi --condizione onset
    uv run python scripts/cv_task2_stack.py debolezze --condizione onset
    uv run python scripts/cv_task2_stack.py primo --condizione ongoing --semi 6 --interni lgb
    uv run python scripts/cv_task2_stack.py mondo_righe --condizione onset --mondi 0 1
    uv run python scripts/cv_task2_stack.py mondo --condizione onset --mondi 0 1
    uv run python scripts/cv_task2_stack.py sintesi --condizione onset --suffisso _mondo0
    uv run python scripts/cv_task2_stack.py mondo_righe --condizione onset --mondi -1 0 1 --e1
    uv run python scripts/cv_task2_stack.py mondo --condizione onset --mondi -1 0 1 --e1
    uv run python scripts/cv_task2_stack.py mondo_righe --condizione ongoing --mondi 0 --insiemi casuali0 attuali ufficiali [--e1]
    uv run python scripts/cv_task2_stack.py mondo_lgb --condizione ongoing --nomi mondo0 mondo0_e1 e1 --insiemi casuali0 attuali ufficiali
    uv run python scripts/cv_task2_stack.py sintesi --condizione ongoing --suffisso _lgb_mondo0

Le ultime tre: gli stessi modelli dei fold previsti su blocchi in mondi spostati (righe_mondo).
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
from cv_task2_decoder import META, almeno_uno, percorso
from tfb.finestre import ORIZZONTE
from tfb.task2_modello import contesto as contesto_finestra

warnings.filterwarnings("ignore", category=RuntimeWarning)
CONDIZIONI = {"onset": "queue_onset", "ongoing": "queue_ongoing"}
SOGLIA = {"onset": 0.40, "ongoing": 0.45}
CURVA = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
VALUTA = {"onset": ["attuali", "ufficiali"], "ongoing": ["casuali0", "casuali1", "casuali2", "attuali", "ufficiali"]}
RICOSTRUITE = {"onset": "attuali", "ongoing": "casuali"}
ALBERI = 400
DIAGNOSI = ["frequenza", "r_ultimo", "in_coda", "istante"]
SD_MONDO = 1.2  # km/h, scarto dei livelli di flusso libero per tratto tra il train e i mesi valutati
XGB = dict(objective="binary:logistic", eta=0.05, max_depth=8, min_child_weight=5, subsample=0.8,
           colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", max_bin=256, seed=0)
CAT = dict(iterations=1000, depth=7, learning_rate=0.05, l2_leaf_reg=10, loss_function="Logloss",
           random_seed=0, verbose=0, allow_writing_files=False)


def allenamento(cond: str, semi: int) -> list[str]:
    return ["attuali"] if cond == "onset" else ["attuali"] + [f"casuali{s}" for s in range(semi)]


def file_probabilita(cond: str):
    return USCITA / f"stack_{cond}.parquet"


def carica(cond: str, semi: int) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Caratteristiche (float32) e descrizione delle righe della condizione, su allenamento e valutazione."""
    allena = allenamento(cond, semi)
    nomi = list(dict.fromkeys(allena + VALUTA[cond]))
    X, info = [], []
    colonne = None
    for i, n in enumerate(nomi):
        d = pd.read_parquet(percorso(n, "righe"))
        d = d[d.condition == CONDIZIONI[cond]].reset_index(drop=True)
        colonne = colonne or [c for c in d.columns if c not in META]
        X.append(d[colonne].to_numpy(np.float32))
        x = d[["fid", "l", "blocco", "ele", "y"] + DIAGNOSI].copy()
        x["passo"] = d.passo.to_numpy().astype(np.int8)
        x["insieme"], x["n_insieme"], x["allena"] = n, i, n in allena
        info.append(x)
        print(f"{n}: {len(d):,} righe", flush=True)
        del d
    info = pd.concat(info, ignore_index=True)
    info["gid"] = pd.factorize(info.n_insieme.astype(np.int64) * 1_000_000 + info.fid)[0]
    return np.concatenate(X), info, colonne


def motore(nome: str, thread: int):
    """(allena, prevedi) per il motore indicato."""
    if nome == "lgb":
        return (lambda X, y: lgb.train(dict(PARAMETRI, num_threads=thread), lgb.Dataset(X, y), num_boost_round=ALBERI),
                lambda m, X: m.predict(X, num_threads=thread))
    if nome == "xgb":
        import xgboost as xgb
        return (lambda X, y: xgb.train(dict(XGB, nthread=thread), xgb.QuantileDMatrix(X, y, nthread=thread), num_boost_round=ALBERI),
                lambda m, X: m.inplace_predict(X))
    from catboost import CatBoostClassifier
    return (lambda X, y: CatBoostClassifier(**CAT, thread_count=thread).fit(X, y),
            lambda m, X: m.predict_proba(X, thread_count=thread)[:, 1])


def primo(cond: str, semi: int, thread: int, motori: list[str], interni: list[str]) -> None:
    """p_{motore}: fuori campione dal fold esterno; i_{motore}{m}: modello sul solo blocco m, sulle righe di allenamento degli altri."""
    t0 = time.time()
    X, info, _ = carica(cond, semi)
    f = file_probabilita(cond)
    vecchio = pd.read_parquet(f) if f.exists() else None
    b, al, ele, y = info.blocco.to_numpy(), info.allena.to_numpy(), info.ele.to_numpy(), info.y.to_numpy()
    for nome in motori:
        if vecchio is not None and f"p_{nome}" in vecchio:
            info[f"p_{nome}"] = vecchio[f"p_{nome}"].to_numpy()
            continue
        fit, prevedi = motore(nome, thread)
        p = np.full(len(info), np.nan, np.float32)
        for k in range(3):
            tr = al & ele & (b != k)
            m = fit(X[tr], y[tr])
            p[b == k] = prevedi(m, X[b == k])
            print(f"{cond} {nome} fold {k}: {tr.sum():,} righe ({time.time() - t0:.0f} s)", flush=True)
        info[f"p_{nome}"] = p
    for nome in interni:
        if vecchio is not None and f"i_{nome}0" in vecchio:
            for m in range(3):
                info[f"i_{nome}{m}"] = vecchio[f"i_{nome}{m}"].to_numpy()
            continue
        fit, prevedi = motore(nome, thread)
        for m in range(3):
            p = np.full(len(info), np.nan, np.float32)
            tr, te = al & ele & (b == m), al & (b != m)
            p[te] = prevedi(fit(X[tr], y[tr]), X[te])
            info[f"i_{nome}{m}"] = p
            print(f"{cond} {nome} interno {m} ({time.time() - t0:.0f} s)", flush=True)
    if vecchio is not None:
        for c in vecchio.columns:
            if c.startswith(("s_",)) and c not in info:
                info[c] = vecchio[c].to_numpy()
    info.drop(columns=["gid"]).to_parquet(f, index=False)


def contesto(p: np.ndarray, info: pd.DataFrame, onset: bool) -> tuple[np.ndarray, list[str]]:
    """Contesto della probabilità p nella sua finestra (task2_modello.contesto, lo stesso dell'invio)."""
    L = (info.groupby("gid").l.max() + 1).sort_index().to_numpy()
    return contesto_finestra(p, info.gid.to_numpy(), info.passo.to_numpy(), info.l.to_numpy(), L, onset)


def secondo(cond: str, semi: int, thread: int, varianti: list[str]) -> None:
    """Stack annidati: s_{variante} fuori campione dal fold esterno."""
    t0 = time.time()
    X, info, _ = carica(cond, semi)
    pr = pd.read_parquet(file_probabilita(cond))
    b, al, ele, y = info.blocco.to_numpy(), info.allena.to_numpy(), info.ele.to_numpy(), info.y.to_numpy()
    fit, prevedi = motore("lgb", thread)
    for v in varianti:
        motori = ["lgb", "xgb", "cat"] if v == "stack_motori" else ["lgb"]
        out = np.full(len(info), np.nan, np.float32)
        for k in range(3):
            p1 = {}
            for mo in motori:
                interno = np.full(len(info), np.nan, np.float32)
                for j in range(3):
                    if j != k:
                        r = b == j
                        interno[r] = pr[f"i_{mo}{3 - k - j}"].to_numpy()[r]
                p1[mo] = np.where(b == k, pr[f"p_{mo}"].to_numpy(), interno).astype(np.float32)
            extra = [np.column_stack([p1[mo] for mo in motori])]
            if v != "stack":
                extra.append(contesto(p1["lgb"], info, cond == "onset")[0])
            Z = np.hstack([X] + extra)
            tr = al & ele & (b != k)
            assert np.isfinite(Z[tr, X.shape[1]]).all() and np.isfinite(Z[b == k, X.shape[1]]).all()
            out[b == k] = prevedi(fit(Z[tr], y[tr]), Z[b == k])
            print(f"{cond} {v} fold {k}: {Z.shape[1]} caratteristiche ({time.time() - t0:.0f} s)", flush=True)
            del Z
        pr[f"s_{v}"] = out
    pr.to_parquet(file_probabilita(cond), index=False)


def metodi(pr: pd.DataFrame) -> dict[str, np.ndarray]:
    m = {c[2:]: pr[c].to_numpy() for c in pr.columns if c.startswith(("p_", "s_"))}
    for a, bb in (("lgb", "xgb"), ("lgb", "cat")):
        if a in m and bb in m:
            m[f"{a}+{bb}"] = (m[a] + m[bb]) / 2
    if {"lgb", "xgb", "cat"} <= set(m):
        m["lgb+xgb+cat"] = (m["lgb"] + m["xgb"] + m["cat"]) / 3
    return m


def valuta(cond: str, suffisso: str = "") -> pd.DataFrame:
    """IoU per finestra, metodo e soglia sulle finestre di valutazione."""
    pr = pd.read_parquet(USCITA / f"stack_{cond}{suffisso}.parquet")
    tutti = metodi(pr)
    ris = []
    for n in VALUTA[cond]:
        fin = pickle.loads(percorso(n, "finestre").read_bytes())
        r = (pr.insieme == n).to_numpy()
        if not r.any():
            continue
        x = pr.loc[r, ["fid", "passo", "l"]]
        fid = x.fid.to_numpy()
        tagli = np.flatnonzero(np.diff(fid)) + 1
        inizio = np.r_[0, tagli]
        passo, l = x.passo.to_numpy().astype(int) - 1, x.l.to_numpy()
        pm = {k: v[r] for k, v in tutti.items()}
        for a, z in zip(inizio, np.r_[tagli, len(fid)]):
            f = fin[int(fid[a])]
            cand = np.zeros(f["vero"].shape, bool)
            cand[ORIZZONTE - 1 if cond == "onset" else slice(None)] = True
            for nome, v in pm.items():
                P = np.zeros(f["vero"].shape)
                P[passo[a:z], l[a:z]] = v[a:z]
                Pc = np.where(cand, P, -1)
                for s in CURVA:
                    ris.append((n, f["pannello"], f["g"], f["t"], f["blocco"], nome, s,
                                iou_finestra(almeno_uno((P >= s) & cand, Pc), f)))
    return pd.DataFrame(ris, columns=["insieme", "pannello", "g", "t", "fold", "metodo", "soglia", "iou"])


def per_fold(y: pd.DataFrame) -> pd.DataFrame:
    t = y.groupby(["metodo", "fold", "pannello"]).iou.mean().groupby(["metodo", "fold"]).mean().unstack("fold")
    t["media"] = t.mean(axis=1)
    t["min_fold"] = t[[c for c in t.columns if c != "media"]].min(axis=1)
    return t


def boot(y: pd.DataFrame, nuovo: str, rif: str = "lgb", n: int = 1000) -> tuple[float, float, float]:
    """Δ nuovo − rif con bootstrap per corridoio·giorno, media per corridoio e poi sui corridoi."""
    w = y[y.metodo.isin([nuovo, rif])].pivot_table(index=["pannello", "g", "t", "insieme"], columns="metodo", values="iou").reset_index()
    idx = {k: v.to_numpy() for k, v in w.groupby(["pannello", "g"]).groups.items()}
    chiavi = list(idx)
    rng = np.random.default_rng(zlib.crc32(b"stack"))
    camp = []
    for _ in range(n):
        righe = np.concatenate([idx[chiavi[i]] for i in rng.integers(0, len(chiavi), len(chiavi))])
        m = w.loc[righe].groupby("pannello")[[nuovo, rif]].mean()
        camp.append(float((m[nuovo] - m[rif]).mean()))
    return float(np.mean(camp)), float(np.percentile(camp, 2.5)), float(np.percentile(camp, 97.5))


def sintesi(cond: str, suffisso: str = "") -> None:
    r = valuta(cond, suffisso)
    r.to_parquet(USCITA / f"stack_{cond}{suffisso}_finestre.parquet", index=False)
    r["valutazione"] = np.where(r.insieme.str.startswith("casuali"), "casuali", r.insieme)
    pd.set_option("display.width", 220)
    for val in dict.fromkeys([RICOSTRUITE[cond]] + [v for v in r.valutazione.unique()]):
        y = r[(r.valutazione == val) & (r.soglia == SOGLIA[cond])]
        t = per_fold(y)
        d = t.sub(t.loc["lgb"], axis=1)
        print(f"\n{cond}, {val} ({y[y.metodo == 'lgb'].shape[0]} finestre), soglia {SOGLIA[cond]:.2f}: IoU per fold e Δ su lgb")
        righe = []
        for m in t.index:
            riga = {**{f"fold{c}": t.loc[m, c] for c in (0, 1, 2) if c in t}, "media": t.loc[m, "media"], "min_fold": t.loc[m, "min_fold"],
                    "Δ media": d.loc[m, "media"], "Δ min_fold": d.loc[m, "min_fold"]}
            if m != "lgb" and val != "ufficiali":
                riga["boot"], riga["boot_2.5"], riga["boot_97.5"] = boot(y, m)
            righe.append(pd.Series(riga, name=m))
        print(pd.DataFrame(righe).round(4).to_string())
    y = r[r.valutazione == RICOSTRUITE[cond]]
    c = y.groupby(["metodo", "soglia"]).apply(lambda z: per_fold(z.assign(metodo="x")).loc["x", ["media", "min_fold"]]).unstack("soglia")
    print(f"\n{cond}, {RICOSTRUITE[cond]}: curva delle soglie (media | fold peggiore)\n" + c.round(4).to_string())


def nome_variante(seme: int | None, e1: bool) -> str:
    return "_".join(([f"mondo{seme}"] if seme is not None else []) + (["e1"] if e1 else [])) or "train"


def file_righe(variante: str, cond: str, n: str):
    return USCITA / (f"stack_{variante}_righe_{n}.parquet" if cond == "onset" else f"stack_{variante}_{cond}_righe_{n}.parquet")


def finestre_insieme(n: str, cond: str) -> pd.DataFrame:
    """Finestre dell'insieme n della condizione, nell'ordine delle righe di cv_task2_decoder.py."""
    from cv_task2_selezione import ufficiali
    from tfb.finestre import ongoing_casuali
    if n == "attuali":
        x = pd.read_parquet(USCITA / "finestre_train.parquet")[["pannello", "g", "t", "condition"]]
    elif n == "ufficiali":
        x = ufficiali()
    else:
        x = pd.concat([ongoing_casuali(p, int(n.removeprefix("casuali"))) for p in PANNELLI], ignore_index=True)
    return x[x.condition == CONDIZIONI[cond]]


def righe_mondo(seme: int | None, e1: bool = False, cond: str = "onset", insiemi: list[str] | None = None) -> None:
    """Righe di valutazione in un mondo spostato: velocità di ogni tratto + δ ~ N(0; 1,2 km/h).

    Come cv_mondo.sposta nel Task 1: il peso va da 0 sotto 0,8·v_f a 1 sopra 0,95·v_f, sullo stato vero
    della cella, per storia e riga a T e anche per lo strato mascherato, da cui E1 stima lo scarto.
    Le etichette di coda (sotto 0,6·v_f) non cambiano. seme None: mondo del train; e1: righe con E1.
    """
    import dataclasses

    import cv_task2
    from tfb.dati import tratti
    originale = cv_task2.tensori
    spost = {}

    def spostati(p: str, split: str, strato: str):
        t = originale(p, split, strato)
        if seme is None:
            return t
        if p not in spost:
            vf = tratti(p).free_speed_kmh.to_numpy(np.float32)
            w = np.clip((originale(p, split, "vero").speed / vf[None, None, :] - 0.80) / 0.15, 0.0, 1.0)
            delta = np.random.default_rng(zlib.crc32(f"{p}:{seme}".encode())).normal(0.0, SD_MONDO, len(vf))
            spost[p] = np.nan_to_num(delta[None, None, :] * w).astype(np.float32)
        return dataclasses.replace(t, speed=(t.speed + spost[p]).astype(t.speed.dtype))

    cv_task2.tensori = spostati
    try:
        for n in insiemi or VALUTA[cond]:
            d, _ = cv_task2.costruisci_righe(finestre_insieme(n, cond), e1=e1)
            d.to_parquet(file_righe(nome_variante(seme, e1), cond, n), index=False)
            print(f"{nome_variante(seme, e1)} {cond} {n}: {len(d):,} righe", flush=True)
    finally:
        cv_task2.tensori = originale


def mondo(semi_mondo: list[str], thread: int) -> None:
    """Onset: gli stessi modelli dei fold (allenati nel mondo del train) previsti sui blocchi valutati spostati.

    semi_mondo: nomi delle varianti di righe (nome_variante: mondo0, mondo0_e1, e1, …).
    """
    t0 = time.time()
    X, info, colonne = carica("onset", 6)
    pr = pd.read_parquet(file_probabilita("onset"))
    b, al, ele, y = info.blocco.to_numpy(), info.allena.to_numpy(), info.ele.to_numpy(), info.y.to_numpy()
    Xm = {}
    for s in semi_mondo:
        parti = []
        for n in VALUTA["onset"]:
            d = pd.read_parquet(USCITA / f"stack_{s}_righe_{n}.parquet")
            r = (info.insieme == n).to_numpy()
            assert (d.l.to_numpy() == info.l.to_numpy()[r]).all() and (d.y.to_numpy() == info.y.to_numpy()[r]).all()
            parti.append(d[colonne].to_numpy(np.float32))
        Xm[s] = np.concatenate(parti)
    out = {s: info.drop(columns=["gid"]).copy() for s in semi_mondo}
    for k in range(3):
        tr, te = al & ele & (b != k), b == k
        p1, pm = {}, {s: {} for s in semi_mondo}
        for mo in ("lgb", "xgb", "cat"):
            fit, prevedi = motore(mo, thread)
            m = fit(X[tr], y[tr])
            assert np.abs(prevedi(m, X[te]) - pr[f"p_{mo}"].to_numpy()[te]).max() < 1e-5
            interno = np.full(len(info), np.nan, np.float32)
            for j in range(3):
                if j != k:
                    interno[b == j] = pr[f"i_{mo}{3 - k - j}"].to_numpy()[b == j]
            p1[mo] = np.where(te, pr[f"p_{mo}"].to_numpy(), interno).astype(np.float32)
            for s in semi_mondo:
                pm[s][mo] = np.where(te, np.nan, p1[mo]).astype(np.float32)
                pm[s][mo][te] = prevedi(m, Xm[s][te])
                out[s].loc[te, f"p_{mo}"] = pm[s][mo][te]
        fit, prevedi = motore("lgb", thread)
        for v, motori in (("stack_ctx", ["lgb"]), ("stack_motori", ["lgb", "xgb", "cat"])):
            Z = np.hstack([X, np.column_stack([p1[mo] for mo in motori]), contesto(p1["lgb"], info, True)[0]])
            m = fit(Z[tr], y[tr])
            for s in semi_mondo:
                Zs = np.hstack([Xm[s], np.column_stack([pm[s][mo] for mo in motori]), contesto(pm[s]["lgb"], info, True)[0]])
                out[s].loc[te, f"s_{v}"] = prevedi(m, Zs[te])
        print(f"mondo fold {k} ({time.time() - t0:.0f} s)", flush=True)
    for s in semi_mondo:
        out[s].to_parquet(USCITA / f"stack_onset_{s}.parquet", index=False)


def mondo_lgb(cond: str, varianti: list[str], insiemi: list[str], semi: int, thread: int, motori: list[str]) -> None:
    """Motori singoli (LightGBM con la ricetta di v15, XGBoost): modelli dei fold previsti sulle righe originali e su quelle delle varianti."""
    t0 = time.time()
    X, info, colonne = carica(cond, semi)
    b, al, ele, y = info.blocco.to_numpy(), info.allena.to_numpy(), info.ele.to_numpy(), info.y.to_numpy()
    tieni = info.insieme.isin(insiemi).to_numpy()
    sotto = info.loc[tieni]
    Xv = {}  # solo le righe degli insiemi valutati, nell'ordine di info
    for v in varianti:
        Xv[v] = np.full((tieni.sum(), X.shape[1]), np.nan, np.float32)
        for n in insiemi:
            d = pd.read_parquet(file_righe(v, cond, n))
            r = (sotto.insieme == n).to_numpy()
            assert (d.l.to_numpy() == sotto.l.to_numpy()[r]).all() and (d.y.to_numpy() == sotto.y.to_numpy()[r]).all()
            Xv[v][r] = d[colonne].to_numpy(np.float32)
            del d
    out = {v: sotto.drop(columns=["gid"]).copy() for v in ["train"] + varianti}
    for mo in motori:
        fit, prevedi = motore(mo, thread)
        for k in range(3):
            tr, te = al & ele & (b != k), b == k
            m = fit(X[tr], y[tr])
            sel = te[tieni]
            out["train"].loc[out["train"].index[sel], f"p_{mo}"] = prevedi(m, X[te & tieni])
            for v in varianti:
                out[v].loc[out[v].index[sel], f"p_{mo}"] = prevedi(m, Xv[v][sel])
            print(f"{cond} mondo {mo} fold {k} ({time.time() - t0:.0f} s)", flush=True)
    for v, o in out.items():
        o.to_parquet(USCITA / f"stack_{cond}_lgb_{v}.parquet", index=False)


def fasce(x: pd.Series, tagli: list[float], etichette: list[str]) -> pd.Series:
    return pd.cut(x, [-np.inf] + tagli + [np.inf], labels=etichette, right=False)


def debolezze(cond: str) -> None:
    """IoU dei motori per fasce di finestra; per l'onset errori per cella di LightGBM per fasce di frequenza e velocità."""
    pr = pd.read_parquet(file_probabilita(cond))
    r = pd.read_parquet(USCITA / f"stack_{cond}_finestre.parquet")
    r = r[r.insieme.str.startswith(RICOSTRUITE[cond]) & (r.soglia == SOGLIA[cond])]
    # descrizione delle finestre dalla verità: celle in coda e frequenza storica media dei tratti in coda
    desc = []
    for n in r.insieme.unique():
        fin = pickle.loads(percorso(n, "finestre").read_bytes())
        x = pr[pr.insieme == n]
        ultimo = x[x.passo == ORIZZONTE]
        fr = {fid: g.set_index("l").frequenza for fid, g in ultimo.groupby("fid")}
        for f in (f for f in fin if f["condition"] == CONDIZIONI[cond]):
            vero = f["vero"] & f["ele"]
            riga = vero[-1] if cond == "onset" else vero.any(axis=0)
            fq = fr[f["fid"]].reindex(np.flatnonzero(riga)).mean() if riga.any() else np.nan
            desc.append((n, f["pannello"], f["g"], f["t"], int(vero[-1].sum() if cond == "onset" else vero.sum()), fq, f["t"] * 5 // 60))
    desc = pd.DataFrame(desc, columns=["insieme", "pannello", "g", "t", "celle_vere", "freq_vere", "ora"])
    r = r.merge(desc, on=["insieme", "pannello", "g", "t"])
    r["fascia_celle"] = fasce(r.celle_vere, [1, 2, 4, 8, 16], ["0", "1", "2–3", "4–7", "8–15", "16+"])
    r["fascia_freq"] = fasce(r.freq_vere, [0.02, 0.05, 0.1, 0.2, 0.4], ["<0,02", "0,02–0,05", "0,05–0,1", "0,1–0,2", "0,2–0,4", "≥0,4"])
    r["fascia_ora"] = fasce(r.ora, [6, 10, 15, 20], ["0–6", "6–10", "10–15", "15–20", "20–24"])
    pd.set_option("display.width", 220)
    for col in ("pannello", "fascia_ora", "fascia_celle", "fascia_freq"):
        t = r.groupby([col, "metodo"], observed=True).iou.mean().unstack("metodo")
        t.insert(0, "finestre", r[r.metodo == "lgb"].groupby(col, observed=True).size())
        t.insert(1, "per fold (lgb)", r[r.metodo == "lgb"].groupby([col, "fold"], observed=True).iou.mean().unstack("fold").round(3).astype(str).agg(" ".join, axis=1))
        print(f"\n{cond}: IoU medio per {col} (media semplice sulle finestre)\n" + t.round(3).to_string())
    if cond != "onset":
        return
    x = pr[pr.insieme == "attuali"].copy()
    x = x[x.ele]
    pred = x.p_lgb.to_numpy() >= SOGLIA[cond]
    x["errore"] = np.select([pred & (x.y == 0), ~pred & (x.y == 1), pred & (x.y == 1)], ["falso positivo", "falso negativo", "giusto"], "negativo giusto")
    x["fascia_freq"] = fasce(x.frequenza, [0.02, 0.05, 0.1, 0.2, 0.4], ["<0,02", "0,02–0,05", "0,05–0,1", "0,1–0,2", "0,2–0,4", "≥0,4"])
    x["fascia_r"] = fasce(x.r_ultimo, [0.7, 0.8, 0.9, 1.0], ["<0,7", "0,7–0,8", "0,8–0,9", "0,9–1", "≥1"])
    for col in ("fascia_freq", "fascia_r"):
        t = x.groupby([col, "errore"], observed=True).size().unstack("errore").fillna(0).astype(int)
        t["quota in coda"] = (x.groupby(col, observed=True).y.mean()).round(4)
        print(f"\nonset, celle a T+30 delle attuali (soglia {SOGLIA[cond]:.2f}, LightGBM, senza «almeno una»): {col}\n" + t.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=["primo", "secondo", "sintesi", "debolezze", "mondo_righe", "mondo", "mondo_lgb"])
    ap.add_argument("--condizione", choices=list(CONDIZIONI), required=True)
    ap.add_argument("--semi", type=int, default=6)
    ap.add_argument("--thread", type=int, default=14)
    ap.add_argument("--motori", nargs="+", default=["lgb", "xgb", "cat"])
    ap.add_argument("--interni", nargs="*", default=["lgb", "xgb", "cat"])
    ap.add_argument("--varianti", nargs="+", default=["stack", "stack_ctx", "stack_motori"])
    ap.add_argument("--mondi", type=int, nargs="+", default=[0, 1], help="semi dei mondi spostati; −1 = mondo del train")
    ap.add_argument("--e1", action="store_true", help="mondo_righe: righe con E1; mondo: usa le righe con E1")
    ap.add_argument("--insiemi", nargs="+", help="mondo_righe, mondo_lgb: insiemi di valutazione (default: tutti)")
    ap.add_argument("--nomi", nargs="+", help="mondo_lgb: varianti di righe (mondo0, mondo0_e1, e1, …)")
    ap.add_argument("--suffisso", default="", help="sintesi di stack_<condizione><suffisso>.parquet, per esempio _mondo0")
    a = ap.parse_args()
    if a.fase == "primo":
        primo(a.condizione, a.semi, a.thread, a.motori, a.interni)
    elif a.fase == "secondo":
        secondo(a.condizione, a.semi, a.thread, a.varianti)
    elif a.fase == "sintesi":
        sintesi(a.condizione, a.suffisso)
    elif a.fase == "mondo_righe":
        for s in a.mondi:
            righe_mondo(None if s < 0 else s, a.e1, a.condizione, a.insiemi)
    elif a.fase == "mondo":
        mondo([nome_variante(None if s < 0 else s, a.e1) for s in a.mondi], a.thread)
    elif a.fase == "mondo_lgb":
        mondo_lgb(a.condizione, a.nomi, a.insiemi or VALUTA[a.condizione], a.semi, a.thread, a.motori)
    else:
        debolezze(a.condizione)
