"""Task 1: ricostruzione di velocità e flusso nelle celle oscurate."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .dati import Tensori, tensori


def evidenza(t: Tensori) -> tuple[np.ndarray, np.ndarray]:
    """Velocità e flusso osservati ed eleggibili; NaN altrove.

    Le celle non eleggibili hanno pct_observed sotto 75 e non sono verità per
    nessun compito: non si usano come evidenza.
    """
    ok = t.eleggibile & np.isfinite(t.speed) & np.isfinite(t.flow)
    return np.where(ok, t.speed, np.nan), np.where(ok, t.flow, np.nan)


def interpola(x: np.ndarray, asse: int) -> np.ndarray:
    """Interpolazione lineare dei NaN lungo un asse; agli estremi ripete il valore più vicino.

    Le righe senza alcun valore restano NaN.
    """
    x = np.moveaxis(x, asse, -1)
    forma = x.shape
    y = x.reshape(-1, forma[-1]).copy()
    idx = np.arange(forma[-1])
    for i in np.flatnonzero(np.isnan(y).any(axis=1)):
        r = y[i]
        ok = ~np.isnan(r)
        if ok.any():
            r[~ok] = np.interp(idx[~ok], idx[ok], r[ok])
    return np.moveaxis(y.reshape(forma), -1, asse)


@lru_cache(maxsize=2)
def profilo(pannello: str) -> tuple[np.ndarray, np.ndarray]:
    """Mediana per istante del giorno e tratto sul train completo: (288, L) per v e q."""
    v = tensori(pannello, "train", "vero")
    vs, qs = evidenza(v)
    return np.nanmedian(vs, axis=0), np.nanmedian(qs, axis=0)


@lru_cache(maxsize=2)
def _modelli_finali(nome: str = "finale"):
    import json

    import lightgbm as lgb

    from . import dati
    c = dati.MODELLI / "task1"
    colonne = json.loads((c / f"{nome}_parametri.json").read_text())["colonne"]
    return {n: lgb.Booster(model_file=str(c / f"{nome}_{n}.txt")) for n in ("v", "q")}, colonne


def finestre_split(t: Tensori):
    """Finestre del Task 2 dello split come (g, t) dell'origine; vuoto per i corridoi senza Task 2."""
    import pandas as pd

    from .dati import RILASCIO
    if not (RILASCIO / "task2" / t.pannello).exists():  # corridoio senza Task 2: nessun buco
        return pd.DataFrame({"g": np.zeros(0, int), "t": np.zeros(0, int)})
    # se il corridoio ha il Task 2 il file dello split deve esserci: un ripiego muto qui
    # toglierebbe il modello sui buchi senza avvisare
    w = pd.read_csv(RILASCIO / "task2" / t.pannello / t.split / "window_index.csv")
    gi = {d: i for i, d in enumerate(t.date)}
    origine = pd.to_datetime(w.forecast_origin)
    return pd.DataFrame({"g": w.date.map(gi).to_numpy(int), "t": ((origine.dt.hour * 60 + origine.dt.minute) // 5).to_numpy(int)})


def regione_finestre(t: Tensori, prima: int = 0, dopo: int = 24) -> np.ndarray:
    """(D, 288, L) bool: istanti da T+prima a T+dopo attorno a ogni finestra del Task 2 dello split.

    Nei mesi valutati, dopo ogni origine T il corridoio è oscurato da T+1 a T+18 nelle celle non
    bersaglio (MASK_SPEC.md): attorno a T i bersagli hanno un contesto che nel train non c'è.
    """
    out = np.zeros(t.speed.shape, bool)
    for g, T in zip(*finestre_split(t)[["g", "t"]].to_numpy().T):
        out[g, max(0, T + prima):T + dopo + 1, :] = True
    return out


@lru_cache(maxsize=1)
def _rete_buchi():
    from . import dati
    from .rete import carica_rete
    return carica_rete(dati.MODELLI / "task1" / "rete_buchi.pt")


def ricostruisci_lgb(t: Tensori) -> tuple[np.ndarray, np.ndarray]:
    """LightGBM sui residui dell'interpolazione nelle celle bersaglio; v1 altrove (non valutato)."""
    from .caratteristiche import E5, MESE, base, calcola
    from .dati import bersagli_task1, tratti

    v, q = ricostruisci_v1(t)
    modelli, colonne = _modelli_finali()
    b = bersagli_task1(t.pannello, t.split, t)
    X = calcola(t, b, {"R1": 0.2, "R2": 0.3, "R3": 0.5}, mese=bool(set(MESE) & set(colonne)), e5=bool(set(E5) & set(colonne)))
    corsie = tratti(t.pannello).lanes.to_numpy(np.float32)
    g, ti, li = b.g.to_numpy(), b.t.to_numpy(), b.l.to_numpy()
    bv, bq = base(X, "v"), base(X, "q")
    # dove manca anche lo spazio resta la ricostruzione v1 (profilo del train)
    bv = np.where(np.isfinite(bv), bv, v[g, ti, li])
    bq = np.where(np.isfinite(bq), bq, q[g, ti, li] / corsie[li])
    v_hat = bv + modelli["v"].predict(X[colonne], num_threads=8)
    q_hat = (bq + modelli["q"].predict(X[colonne], num_threads=8)) * corsie[li]
    v[g, ti, li] = np.clip(v_hat, 3.0, None)
    q[g, ti, li] = np.clip(q_hat, 50.0, None)
    return v, q


def ricostruisci_lgb_buchi(t: Tensori, rete: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Come ricostruisci_lgb; nei bersagli da T a T+24 delle finestre del Task 2 usa il modello
    allenato anche sui buchi del Task 2 riprodotti nel train (scripts/cv_buchi.py).

    Con `rete` lì la previsione è la media di quel modello e della rete di tfb.rete (E2,
    scripts/e2_sequenze.py), come il ramo media0 di cv_pseudomesi.py."""
    from .caratteristiche import MESE, base, calcola
    from .dati import bersagli_task1, tratti

    v, q = ricostruisci_lgb(t)
    b = bersagli_task1(t.pannello, t.split, t)
    g, ti, li = b.g.to_numpy(), b.t.to_numpy(), b.l.to_numpy()
    dentro = regione_finestre(t)[g, ti, li]
    if not dentro.any():
        return v, q
    v1, q1 = ricostruisci_v1(t)
    b = b[dentro].reset_index(drop=True)
    g, ti, li = g[dentro], ti[dentro], li[dentro]
    modelli, colonne = _modelli_finali("finale_buchi")
    X = calcola(t, b, {"R1": 0.2, "R2": 0.3, "R3": 0.5}, mese=bool(set(MESE) & set(colonne)))
    corsie = tratti(t.pannello).lanes.to_numpy(np.float32)
    bv, bq = base(X, "v"), base(X, "q")
    bv = np.where(np.isfinite(bv), bv, v1[g, ti, li])
    bq = np.where(np.isfinite(bq), bq, q1[g, ti, li] / corsie[li])
    v_hat = np.clip(bv + modelli["v"].predict(X[colonne], num_threads=8), 3.0, None)
    q_hat = np.clip((bq + modelli["q"].predict(X[colonne], num_threads=8)) * corsie[li], 50.0, None)
    if rete:
        from .rete import prevedi_regione
        tutti = bersagli_task1(t.pannello, t.split, t)
        bersaglio = np.zeros(t.speed.shape, bool); bersaglio[tutti.g, tutti.t, tutti.l] = True
        vr, qr = prevedi_regione(_rete_buchi(), t, bersaglio, finestre_split(t), v1, q1, g, ti, li)
        if not (np.isfinite(vr).all() and np.isfinite(qr).all()):
            raise ValueError(f"{t.pannello} {t.split}: bersagli della regione senza previsione della rete")
        v_hat = (v_hat + np.clip(vr, 3.0, None)) / 2
        q_hat = (q_hat + np.clip(qr * corsie[li], 50.0, None)) / 2
    v[g, ti, li] = v_hat
    q[g, ti, li] = q_hat
    return v, q


def ricostruisci_lgb_buchi_rete(t: Tensori) -> tuple[np.ndarray, np.ndarray]:
    return ricostruisci_lgb_buchi(t, rete=True)


@lru_cache(maxsize=2)
def livello_train(pannello: str) -> np.ndarray:
    """(L,) velocità di notte di ogni tratto nel mondo del train: il train è un mondo solo (0,06 km/h tra i mesi)."""
    from .caratteristiche import livello_mese
    return np.nanmedian(livello_mese(tensori(pannello, "train")), axis=0)


def peso_libero(v: np.ndarray, vf: np.ndarray) -> np.ndarray:
    """Quota dello scarto di livello che tocca una velocità: piena in flusso libero, nulla in coda.

    Nei mesi valutati lo scarto per tratto è tutto in flusso libero (pendenza 0,99 sullo scarto
    notturno a bassa occupazione) e sparisce in coda (0,16): README, E1.
    """
    return np.clip((v / vf - 0.80) / 0.15, 0.0, 1.0)


def scarto_livello(t: Tensori) -> np.ndarray:
    """(D, L) scarto del livello notturno del mese di ogni giorno da quello del train; 0 dove non si stima."""
    from .caratteristiche import livello_mese
    d = livello_mese(t) - livello_train(t.pannello)[None, :]
    return np.where(np.isfinite(d), d, 0.0).astype(np.float32)


def riporta(t: Tensori, d: np.ndarray, verso: float = -1.0) -> Tensori:
    """Velocità spostate di verso·d (d per giorno e tratto) con il peso del flusso libero."""
    import dataclasses

    from .dati import tratti
    vf = tratti(t.pannello).free_speed_kmh.to_numpy(np.float32)
    v = t.speed + verso * d[:, None, :] * peso_libero(t.speed, vf)
    return dataclasses.replace(t, speed=v.astype(t.speed.dtype))


def equivariante(metodo):
    """E1: `metodo` ricostruisce il mese riportato nel mondo del train, poi alle velocità si restituisce lo scarto.

    Un tratto il cui livello in flusso libero è cambiato rispetto al train non chiede più ai modelli
    di correggerlo: lo scarto si toglie dagli ingressi e si rimette nell'uscita (REVISIONE_AVVERSARIALE_v2 §5).
    """
    def ricostruisci(t: Tensori) -> tuple[np.ndarray, np.ndarray]:
        from .dati import tratti
        d = scarto_livello(t)
        v, q = metodo(riporta(t, d))
        vf = tratti(t.pannello).free_speed_kmh.to_numpy(np.float32)
        return (v + d[:, None, :] * peso_libero(v, vf)).astype(np.float32), q
    ricostruisci.__name__ = f"equivariante_{metodo.__name__}"
    return ricostruisci


def ricostruisci_v1(t: Tensori) -> tuple[np.ndarray, np.ndarray]:
    """Interpolazione nel tempo sullo stesso tratto; poi nello spazio; poi profilo del train."""
    vs, qs = evidenza(t)
    v, q = interpola(vs, 1), interpola(qs, 1)
    vx, qx = interpola(vs, 2), interpola(qs, 2)
    v, q = np.where(np.isnan(v), vx, v), np.where(np.isnan(q), qx, q)
    pv, pq = profilo(t.pannello)
    v = np.where(np.isnan(v), pv[None], v)
    q = np.where(np.isnan(q), pq[None], q)
    return v.astype(np.float32), q.astype(np.float32)


# v11: v10 con E1 (stessi modelli); validato sui mesi veri con rimaschera_valutati.py --e1
ricostruisci_lgb_buchi_e1 = equivariante(ricostruisci_lgb_buchi)
# v14: v13 con la media del modello sui buchi e della rete E2 nella regione delle finestre
ricostruisci_lgb_buchi_rete_e1 = equivariante(ricostruisci_lgb_buchi_rete)
