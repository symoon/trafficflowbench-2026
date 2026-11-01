"""Task 4: flussi per percorso dai conteggi sui tratti e dal prior debole."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from .dati import RILASCIO


def operatore(pannello: str) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Tabella dei percorsi, tratti dell'incidenza e matrice A (tratti × percorsi)."""
    rete = RILASCIO / "corridors" / pannello / "network"
    percorsi = pd.read_csv(rete / "path_set.csv", dtype=str)
    inc = pd.read_csv(rete / "path_link_incidence.csv", dtype=str)
    link_ids = inc.link_id.drop_duplicates().tolist()
    pi = {p: i for i, p in enumerate(percorsi.path_id)}
    li = {l: i for i, l in enumerate(link_ids)}
    A = np.zeros((len(link_ids), len(percorsi)))
    A[inc.link_id.map(li).to_numpy(), inc.path_id.map(pi).to_numpy()] = 1.0
    return percorsi, link_ids, A


def dati_split(pannello: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = RILASCIO / "task4" / pannello / split
    conteggi = pd.read_csv(base / "synthetic_link_counts.csv", dtype={"link_id": str})
    prior = pd.read_csv(base / "synthetic_weak_prior.csv", dtype={"path_id": str})
    return conteggi, prior


def prior_medio(pannello: str) -> pd.Series:
    """Media aritmetica dei prior di train, validation e private, per percorso (variante v5, scartata).

    I tre prior hanno una parte comune (corr dei log 0,73–0,75) e una propria di ogni split
    (σ del log ≈ 0,62). Se la parte propria fosse un moltiplicatore Gamma a media 1, la media
    aritmetica sarebbe la stima di massima verosimiglianza della parte comune. In classifica
    la NNLS su questa media ha reso molto peggio di quella sul prior dello split.
    """
    percorsi, _, _ = operatore(pannello)
    b = [dati_split(pannello, s)[1].set_index("path_id").reindex(percorsi.path_id).path_flow for s in ("train", "validation", "private")]
    return pd.concat(b, axis=1).mean(axis=1)


def nnls_prior(pannello: str, split: str, lam: float = 0.05, medio: bool = False) -> pd.DataFrame:
    """min ||A f - c||² + λ ||f - b||², f ≥ 0, sui soli tratti con conteggio (baseline ufficiale).

    Con medio=True il prior b è la media dei tre prior rilasciati invece di quello dello split.
    """
    percorsi, link_ids, A = operatore(pannello)
    conteggi, prior = dati_split(pannello, split)
    li = {l: i for i, l in enumerate(link_ids)}
    righe = [li[l] for l in conteggi.link_id if l in li]
    c = conteggi.set_index("link_id").loc[[l for l in conteggi.link_id if l in li], "count"].to_numpy(float)
    b = (prior_medio(pannello) if medio else prior.set_index("path_id").path_flow).reindex(percorsi.path_id).fillna(0.0).to_numpy(float)
    M = np.vstack([A[righe], np.sqrt(lam) * np.eye(A.shape[1])])
    y = np.concatenate([c, np.sqrt(lam) * b])
    f, _ = nnls(M, y, maxiter=50 * A.shape[1])
    out = percorsi[["path_id"]].copy()
    out["panel"] = pannello
    out["departure_time"] = str(prior.departure_time.iloc[0])
    out["path_flow"] = f
    return out


def kl_prior(pannello: str, split: str, rho: float = 1e-6) -> pd.DataFrame:
    """min KL(f || b) + ||A f - c||² / (2 ρ c̄), f > 0: il prior si corregge per fattori moltiplicativi.

    Ogni percorso riceve il prodotto dei fattori dei tratti contati che attraversa
    (f = b · exp(Aᵀμ)), quindi una distorsione del prior sulla lunghezza dei viaggi si
    corregge senza azzerare percorsi. Si risolve il duale, liscio e convesso, in μ.
    """
    percorsi, link_ids, A = operatore(pannello)
    conteggi, prior = dati_split(pannello, split)
    li = {l: i for i, l in enumerate(link_ids)}
    tenuti = [l for l in conteggi.link_id if l in li]
    Ac = A[[li[l] for l in tenuti]]
    c = conteggi.set_index("link_id").loc[tenuti, "count"].to_numpy(float)
    b = prior.set_index("path_id").reindex(percorsi.path_id).path_flow.fillna(0.0).to_numpy(float)
    sc = c.mean()

    def duale(mu):
        e = b * np.exp(np.clip(Ac.T @ mu, -50.0, 50.0))
        return (e.sum() - c @ mu + 0.5 * rho * sc * mu @ mu) / sc, (Ac @ e - c + rho * sc * mu) / sc
    r = minimize(duale, np.zeros(len(c)), jac=True, method="L-BFGS-B", options=dict(maxiter=5000, gtol=1e-10))
    f = b * np.exp(np.clip(Ac.T @ r.x, -50.0, 50.0))
    out = percorsi[["path_id"]].copy()
    out["panel"] = pannello
    out["departure_time"] = str(prior.departure_time.iloc[0])
    out["path_flow"] = f
    return out


def nnls_prior_medio(pannello: str, split: str) -> pd.DataFrame:
    return nnls_prior(pannello, split, medio=True)
