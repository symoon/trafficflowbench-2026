"""Costruzione del file d'invio unico a partire dalle previsioni per compito.

Il Task 1 arriva come tensori (giorno, istante, tratto) per ogni corridoio e split,
il Task 2 e il Task 4 come tabelle nelle loro chiavi naturali. La chiave ufficiale
(submission_key.csv) si legge una volta e si tiene in cache come parquet.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .dati import CACHE, RILASCIO, _indice_istante, tensori

COLONNE = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
SPLIT_VALUTATI = ("validation", "private")


def chiave() -> pd.DataFrame:
    percorso = CACHE / "submission_key.parquet"
    if percorso.exists():
        return pd.read_parquet(percorso)
    k = pd.read_csv(
        RILASCIO / "submission_key.csv", dtype=str,
        usecols=["submission_id", "task", "split", "panel", "timestamp", "station_id", "link_id",
                 "mask_regime", "window_id", "departure_time", "path_id"],
    )
    k["submission_id"] = k.submission_id.astype(np.int64)
    for c in ("task", "split", "panel", "mask_regime"):
        k[c] = k[c].astype("category")
    CACHE.mkdir(parents=True, exist_ok=True)
    k.to_parquet(percorso, index=False)
    return k


def scrivi(
    stato: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    code: pd.DataFrame,
    odme: pd.DataFrame,
    uscita: Path,
) -> Path:
    """Scrive l'invio e lo comprime in zip; ritorna il percorso dello zip.

    `stato[(pannello, split)] = (velocità, flusso)`, tensori (D, 288, L) nello stesso
    ordine di `tensori(pannello, split)`. `code` ha window_id, timestamp, link_id,
    queue_pred; `odme` ha panel, departure_time, path_id, path_flow.
    """
    k = chiave()
    out = pd.DataFrame({"submission_id": k.submission_id.to_numpy(), "task": k.task.astype(str).to_numpy()})
    for c in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
        out[c] = np.nan

    s = k.task == "state"
    for (pannello, split), righe in k[s].groupby(["panel", "split"], observed=True):
        v, q = stato[(pannello, split)]
        t = tensori(pannello, split)
        gi = {d: i for i, d in enumerate(t.date)}
        li = {l: i for i, l in enumerate(t.link_id)}
        g = righe.timestamp.str.slice(0, 10).map(gi).to_numpy(np.int64)
        ti = _indice_istante(righe.timestamp)
        l = righe.link_id.map(li).to_numpy(np.int64)
        out.loc[righe.index, "speed_kmh"] = v[g, ti, l].astype(np.float64)
        out.loc[righe.index, "flow_vph"] = q[g, ti, l].astype(np.float64)

    s = k.task == "queue"
    m = k.loc[s, ["window_id", "timestamp", "link_id"]].merge(
        code[["window_id", "timestamp", "link_id", "queue_pred"]], how="left", validate="one_to_one")
    out.loc[s, "queue_pred"] = m.queue_pred.to_numpy(np.float64)

    s = k.task == "odme"
    m = k.loc[s, ["panel", "departure_time", "path_id"]].astype(str).merge(
        odme[["panel", "departure_time", "path_id", "path_flow"]].astype({"panel": str, "path_id": str}),
        how="left", validate="one_to_one")
    out.loc[s, "path_flow"] = m.path_flow.to_numpy(np.float64)

    _controlla(out)
    for c in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
        out[c] = out[c].fillna(0.0)
    out["queue_pred"] = out.queue_pred.astype(np.int8)
    uscita.parent.mkdir(parents=True, exist_ok=True)
    out[COLONNE].to_csv(uscita, index=False, float_format="%.3f")
    zp = uscita.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(uscita, uscita.name)
    return zp


def _controlla(out: pd.DataFrame) -> None:
    """Gli stessi controlli del valutatore, prima di sprecare un invio."""
    per = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}
    for task, colonne in per.items():
        r = out.task == task
        for c in colonne:
            x = out.loc[r, c].to_numpy(float)
            if not np.isfinite(x).all():
                raise ValueError(f"{task}.{c}: {int((~np.isfinite(x)).sum())} valori mancanti o non finiti")
    x = out.loc[out.task == "queue", "queue_pred"]
    if not x.isin([0, 1]).all():
        raise ValueError("queue_pred non binario")
    if (out.loc[out.task == "odme", "path_flow"] < 0).any():
        raise ValueError("path_flow negativo")
    if (out.loc[out.task == "state", "speed_kmh"] <= 0).any():
        raise ValueError("velocità non positive")
    if out.submission_id.duplicated().any():
        raise ValueError("submission_id duplicati")
    # contro il template originale, non contro la chiave in cache: stessi ID, stesso ordine, stessi task
    modello = pd.read_csv(RILASCIO / "sample_submission.csv", usecols=["submission_id", "task"])
    if len(modello) != len(out) or not (modello.submission_id.to_numpy() == out.submission_id.to_numpy()).all():
        raise ValueError("gli ID non coincidono con sample_submission.csv")
    if not (modello.task.astype(str).to_numpy() == out.task.to_numpy()).all():
        raise ValueError("i task non coincidono con sample_submission.csv")
