"""Task 2: tratti in coda da T+5 a T+30.

Regola dei dati: per una finestra con origine T si usano solo osservazioni con
timestamp ≤ T. La storia della finestra copre T-60 … T-5; la riga a T si legge
dallo strato mascherato dello split, che è dato rilasciato a T.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .dati import RILASCIO, _indice_istante, tensori, tratti

SOGLIA = 0.60  # coda: velocità ≤ 0,6 × velocità libera


def finestre(pannello: str, split: str) -> pd.DataFrame:
    p = RILASCIO / "task2" / pannello / split / "window_index.csv"
    return pd.read_csv(p, dtype=str) if p.exists() else pd.DataFrame()


def stato_a_T(pannello: str, split: str) -> pd.DataFrame:
    """Per ogni finestra e tratto: velocità all'ultimo istante disponibile ≤ T, con il suo ritardo."""
    w = finestre(pannello, split)
    if w.empty:
        return pd.DataFrame()
    tr = tratti(pannello)
    h = pd.read_parquet(RILASCIO / "task2" / pannello / split / "window_history.parquet")
    h = h[h.is_score_eligible.astype(bool) & h.speed_kmh.notna()]
    m = tensori(pannello, split)
    gi = {d: i for i, d in enumerate(m.date)}
    righe = []
    for x in w.itertuples():
        g, ti = gi[x.date], int(_indice_istante(pd.Series([x.forecast_origin]))[0])
        v_T = np.where(m.eleggibile[g, ti], m.speed[g, ti], np.nan)
        hh = h[h.window_id == x.window_id].sort_values("timestamp")
        ultimo = hh.groupby("link_id").speed_kmh.last()
        storia = tr.link_id.map(ultimo).to_numpy(float)
        v = np.where(np.isfinite(v_T), v_T, storia)
        righe.append(pd.DataFrame({"window_id": x.window_id, "condition": x.condition,
                                   "link_id": tr.link_id, "v": v,
                                   "v_libera": tr.free_speed_kmh.to_numpy(float)}))
    return pd.concat(righe, ignore_index=True)


def persistenza(pannello: str, split: str) -> pd.DataFrame:
    """Ripete su tutto l'orizzonte lo stato di coda all'ultimo istante noto ≤ T."""
    s = stato_a_T(pannello, split)
    if s.empty:
        return pd.DataFrame(columns=["window_id", "timestamp", "link_id", "queue_pred"])
    s["coda"] = (s.v <= SOGLIA * s.v_libera).astype(np.int8)
    mod = pd.read_csv(RILASCIO / "task2" / pannello / split / "sample_submission_queue.csv",
                      usecols=["window_id", "timestamp", "link_id"], dtype=str)
    mod = mod[mod.window_id.isin(s.window_id)]
    out = mod.merge(s[["window_id", "link_id", "coda"]], on=["window_id", "link_id"], how="left")
    out["queue_pred"] = out.coda.fillna(0).astype(np.int8)
    return out[["window_id", "timestamp", "link_id", "queue_pred"]]


@lru_cache(maxsize=1)
def _modelli_finali():
    import json

    import lightgbm as lgb
    from . import dati
    c = dati.MODELLI / "task2"
    par = json.loads((c / "finale_parametri.json").read_text())
    modelli = {k: lgb.Booster(model_file=str(c / f"finale_{k}.txt")) for k in par["soglie"]}
    return modelli, par["soglie"], par["colonne"]


@lru_cache(maxsize=1)
def _frequenza(pannello: str) -> np.ndarray:
    """Frequenza di coda per istante e tratto su tutti i giorni del train, che precede i mesi valutati."""
    from .finestre import etichette
    from .task2_modello import frequenza_coda
    coda, ele = etichette(pannello)
    return frequenza_coda(coda, ele, np.arange(len(coda)))


def storia_finestre(pannello: str, split: str):
    """Per ogni finestra: riga dell'indice, t di origine e storia (12, L) di v, q, o; NaN dove non eleggibile."""
    from .finestre import STORIA
    w = finestre(pannello, split)
    tr = tratti(pannello)
    col = {l: i for i, l in enumerate(tr.link_id)}
    h = pd.read_parquet(RILASCIO / "task2" / pannello / split / "window_history.parquet")
    h = h[h.is_score_eligible.astype(bool) & h.speed_kmh.notna()]
    per_finestra = dict(tuple(h.groupby("window_id")))
    for x in w.itertuples():
        t = int(_indice_istante(pd.Series([x.forecast_origin]))[0])
        sv, sq, so = (np.full((STORIA, len(tr)), np.nan, np.float32) for _ in range(3))
        hh = per_finestra.get(x.window_id)
        if hh is not None:
            k = _indice_istante(hh.timestamp) - (t - STORIA)
            li = hh.link_id.map(col).to_numpy()
            ok = (k >= 0) & (k < STORIA) & ~pd.isna(li)
            k, li = k[ok], li[ok].astype(int)
            sv[k, li] = hh.speed_kmh.to_numpy()[ok]
            sq[k, li] = hh.flow_vph.to_numpy()[ok]
            so[k, li] = hh.occupancy.to_numpy()[ok]
        yield x, t, sv, sq, so


@lru_cache(maxsize=2)
def _stack(condizione: str):
    """Primo livello (LightGBM, XGBoost, CatBoost) e secondo livello dello stack di una condizione (allena_task2_stack.py)."""
    import json

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from . import dati
    c = dati.MODELLI / "task2"
    breve = condizione.removeprefix("queue_")
    # v16 è nato con il solo onset: parametri senza la condizione nel nome e senza la voce «secondo»
    par = json.loads((c / ("finale_stack_parametri.json" if breve == "onset" else f"finale_stack_{breve}_parametri.json")).read_text())
    x = xgb.Booster()
    x.load_model(str(c / par["file"]["xgb"]))
    cb = CatBoostClassifier()
    cb.load_model(str(c / par["file"]["cat"]))
    return dict(lgb=lgb.Booster(model_file=str(c / par["file"]["lgb"])), xgb=x, cat=cb,
                secondo=lgb.Booster(model_file=str(c / par.get("secondo", f"finale_stack_{breve}_secondo.txt")))), par


def _probabilita_stack(condizione: str, X: pd.DataFrame, L: int) -> np.ndarray:
    """Probabilità per riga di X (passo per passo, L tratti ciascuno): secondo livello su caratteristiche,
    tre probabilità del primo e contesto di quella di LightGBM nella finestra."""
    from .task2_modello import contesto
    m, par = _stack(condizione)
    Z = X[par["colonne"]].to_numpy(np.float32)
    p1 = [m["lgb"].predict(Z, num_threads=8), m["xgb"].inplace_predict(Z), m["cat"].predict_proba(Z, thread_count=8)[:, 1]]
    n = len(Z)
    ctx, nomi = contesto(p1[0].astype(np.float32), np.zeros(n), X.passo.to_numpy(), np.tile(np.arange(L), n // L),
                         np.array([L]), onset=condizione == "queue_onset")
    assert par["colonne_secondo"] == par["colonne"] + ["p1_lgb", "p1_xgb", "p1_cat"] + nomi
    return m["secondo"].predict(np.hstack([Z, np.column_stack(p1).astype(np.float32), ctx]), num_threads=8)


def modello_stack(pannello: str, split: str) -> pd.DataFrame:
    """Come modello, con le probabilità onset dallo stack di tre motori (v16; cv_task2_stack.py, stack_motori)."""
    return modello(pannello, split, stack=("queue_onset",))


def modello_e1_onset_stack_ongoing(pannello: str, split: str) -> pd.DataFrame:
    """Come modello_e1_onset, con le probabilità ongoing dallo stack di tre motori (v18; cv_task2_stack.py, stack_motori)."""
    return modello(pannello, split, stack=("queue_ongoing",), e1=("queue_onset",))


def modello_e1_onset(pannello: str, split: str) -> pd.DataFrame:
    """Come modello, con E1 sulle finestre onset (v17; cv_task2_stack.py mondo --e1), stessi modelli."""
    return modello(pannello, split, e1=("queue_onset",))


def modello(pannello: str, split: str, stack: tuple[str, ...] = (), e1: tuple[str, ...] = ()) -> pd.DataFrame:
    """LightGBM per tratto e passo (onset: solo T+30), soglia scelta a blocchi, almeno un tratto per finestra.

    Per le condizioni in e1, lo scarto del livello notturno del mese di ogni tratto (notti fino a T
    compreso, strato mascherato dello split) da quello del train si toglie dalle velocità di storia e
    di T prima delle caratteristiche: la finestra torna nel mondo del train (E1 del Task 2).
    """
    from .finestre import ORIZZONTE
    from .task2_modello import caratteristiche, insieme, scarto_e1, togli_scarto
    colonne_out = ["window_id", "timestamp", "link_id", "queue_pred"]
    w = finestre(pannello, split)
    if w.empty:
        return pd.DataFrame(columns=colonne_out)
    modelli, soglie, colonne = _modelli_finali()
    tr = tratti(pannello)
    m = tensori(pannello, split)
    okm = m.eleggibile & np.isfinite(m.speed)
    gi = {d: i for i, d in enumerate(m.date)}
    freq = _frequenza(pannello)
    if e1:
        from .task1 import livello_train
        vf = tr.free_speed_kmh.to_numpy(np.float32)
        vs = np.where(okm, m.speed, np.nan)
        mesi = pd.to_datetime(pd.Series(m.date)).dt.to_period("M").to_numpy()
        livello = livello_train(pannello)
    righe = []
    for x, t, sv, sq, so in storia_finestre(pannello, split):
        g = gi[x.date]
        a_T = [np.where(okm[g, t], z[g, t], np.nan) for z in (m.speed, m.flow, m.occ)]
        if x.condition in e1:
            d = scarto_e1(vs, mesi, g, t, livello)
            sv, a_T[0] = togli_scarto(sv, d, vf), togli_scarto(a_T[0], d, vf)
        X = caratteristiche(pannello, sv, sq, so, *a_T, t, freq)
        passi = [ORIZZONTE] if x.condition == "queue_onset" else list(range(1, ORIZZONTE + 1))
        Xs = pd.concat([X.assign(passo=np.float32(s)) for s in passi], ignore_index=True)
        P = np.zeros((ORIZZONTE, len(tr)))
        if x.condition in stack:
            P[np.array(passi) - 1] = _probabilita_stack(x.condition, Xs, len(tr)).reshape(len(passi), -1)
        else:
            P[np.array(passi) - 1] = modelli[x.condition].predict(Xs[colonne], num_threads=8).reshape(len(passi), -1)
        pred = insieme(P.ravel(), soglie[x.condition], almeno_uno=True).reshape(P.shape)
        origine = pd.Timestamp(x.forecast_origin)
        ts = [(origine + pd.Timedelta(minutes=5 * s)).strftime("%Y-%m-%dT%H:%M:%SZ") for s in range(1, ORIZZONTE + 1)]
        righe.append(pd.DataFrame({"window_id": x.window_id, "timestamp": np.repeat(ts, len(tr)),
                                   "link_id": np.tile(tr.link_id.to_numpy(), ORIZZONTE),
                                   "queue_pred": pred.ravel().astype(np.int8)}))
    pr = pd.concat(righe, ignore_index=True)
    mod = pd.read_csv(RILASCIO / "task2" / pannello / split / "sample_submission_queue.csv",
                      usecols=["window_id", "timestamp", "link_id"], dtype=str)
    out = mod.merge(pr, on=["window_id", "timestamp", "link_id"], how="left")
    assert out.queue_pred.notna().all(), f"{pannello} {split}: righe del modello senza previsione"
    out["queue_pred"] = out.queue_pred.astype(np.int8)
    return out[colonne_out]
