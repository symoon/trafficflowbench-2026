"""Lettura del pacchetto TrafficFlowBench in tensori giorno × istante × tratto.

I tratti sono ordinati da monte a valle: milepost crescente per i corridoi N ed E,
decrescente per S e W (lo stesso ordine del valutatore del Task 3, orientato nel
verso di marcia). Ogni corridoio ha un rilevatore per tratto, quindi l'asse dei
tratti vale anche per le stazioni.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RADICE_PROGETTO = Path(__file__).resolve().parents[2]
RILASCIO = RADICE_PROGETTO / "data" / "kaggle_public"
CACHE = RADICE_PROGETTO / "data" / "cache"
# modelli finali (task1/, task2/); genera_invio la sposta sui modelli congelati di una versione
MODELLI = RADICE_PROGETTO / "outputs"
SPLIT = ("train", "validation", "private")
REGIMI = ("R1", "R2", "R3")
ISTANTI = 288  # passi da 5 minuti in un giorno

CANALI = ("speed_kmh", "flow_vph", "occupancy", "density_occ_linear_vehpkm")


def pannelli() -> list[str]:
    cfg = json.loads((RILASCIO / "config" / "corridors.json").read_text())
    return [p["corridor_id"] for p in cfg["panels"]]


def famiglia(pannello: str) -> str:
    return pannello.rsplit("_", 1)[0]


def _verso_valle_crescente(pannello: str) -> bool:
    return pannello.rsplit("_", 1)[1] in ("N", "E")


def tratti(pannello: str) -> pd.DataFrame:
    """Un tratto per riga, da monte a valle, con i parametri del diagramma fondamentale.

    `lanes`, `free_speed_kmh`, `capacity_vph`, `critical_density`, `k_jam` vengono da
    fd_parameters.csv, che è la fonte usata dai valutatori (le corsie di links.csv e
    della topologia non coincidono). Le rampe vengono da ramp_attachment_map.csv.
    """
    rete = RILASCIO / "corridors" / pannello / "network"
    fd = pd.read_csv(rete / "fd_parameters.csv", dtype={"link_id": str})
    topo = pd.read_csv(rete / "lwr_mainline_topology.csv", dtype=str)
    rampe = pd.read_csv(rete / "ramp_attachment_map.csv", dtype=str)
    giorno = next(iter(sorted((RILASCIO / "corridors" / pannello / "train" / "mainline_states").glob("*/*.parquet"))))
    mp = (
        pq.read_table(giorno, columns=["station_id", "link_id", "milepost"]).to_pandas()
        .drop_duplicates("link_id")
    )
    t = mp.merge(fd, on="link_id", how="left", validate="one_to_one")
    t = t.merge(
        topo[["link_id", "has_sensor", "incoming_link_ids", "outgoing_link_ids"]],
        on="link_id", how="left", validate="one_to_one",
    )
    t["has_sensor"] = t.has_sensor.astype(int)
    on = rampe[rampe.ramp_type == "OR"].groupby("nearest_mainline_link_id").ramp_link_id.agg(list)
    off = rampe[rampe.ramp_type == "FR"].groupby("nearest_mainline_link_id").ramp_link_id.agg(list)
    t["rampe_on"] = t.link_id.map(on).apply(lambda x: x if isinstance(x, list) else [])
    t["rampe_off"] = t.link_id.map(off).apply(lambda x: x if isinstance(x, list) else [])
    t = t.sort_values("milepost", ascending=_verso_valle_crescente(pannello)).reset_index(drop=True)
    t["pos_km"] = (t.milepost - t.milepost.iloc[0]).abs()
    return t


def _indice_istante(ts: pd.Series) -> np.ndarray:
    s = ts.astype(str)
    return (s.str.slice(11, 13).astype(np.int16) * 12 + s.str.slice(14, 16).astype(np.int16) // 5).to_numpy()


def _file_giornalieri(pannello: str, split: str, strato: str) -> list[Path]:
    base = RILASCIO / "corridors" / pannello / split
    if strato == "vero":
        return sorted((base / "mainline_states").glob("*/*.parquet"))
    return sorted((base / "mainline_states_masked").glob("*/*.parquet"), key=lambda p: p.name)


@dataclass
class Tensori:
    """Osservazioni di un corridoio in uno split, asse (giorno, istante, tratto)."""

    pannello: str
    split: str
    strato: str          # "vero" (solo train) oppure "mascherato"
    date: np.ndarray     # (D,) testo YYYY-MM-DD
    regime: np.ndarray   # (D,) R1/R2/R3 per giorno, "" sullo strato vero
    link_id: np.ndarray  # (L,) da monte a valle
    speed: np.ndarray    # (D, 288, L) float32, NaN dove manca
    flow: np.ndarray
    occ: np.ndarray
    dens_occ: np.ndarray
    pct: np.ndarray      # (D, 288, L) uint8, pct_observed
    eleggibile: np.ndarray  # (D, 288, L) bool, is_score_eligible


def tensori(pannello: str, split: str, strato: str = "mascherato", cache: bool = True) -> Tensori:
    if strato not in ("vero", "mascherato"):
        raise ValueError(strato)
    if strato == "vero" and split != "train":
        raise ValueError("solo il train rilascia lo strato non mascherato")
    percorso = CACHE / f"{pannello}__{split}__{strato}.npz"
    if cache and percorso.exists():
        z = np.load(percorso, allow_pickle=False)
        return Tensori(pannello, split, strato, **{k: z[k] for k in z.files})

    ordine = tratti(pannello).link_id.to_numpy()
    pos = {l: i for i, l in enumerate(ordine)}
    files = _file_giornalieri(pannello, split, strato)
    D, L = len(files), len(ordine)
    out = {c: np.full((D, ISTANTI, L), np.nan, np.float32) for c in CANALI}
    pct = np.zeros((D, ISTANTI, L), np.uint8)
    ele = np.zeros((D, ISTANTI, L), bool)
    date, regime = [], []
    colonne = ["date", "timestamp", "link_id", *CANALI, "pct_observed", "is_score_eligible"]
    for d, f in enumerate(files):
        tab = pq.read_table(f, columns=colonne + (["mask_regime"] if strato == "mascherato" else []))
        x = tab.to_pandas()
        if x.date.nunique() != 1:
            raise ValueError(f"{f}: più di una data")
        date.append(str(x.date.iloc[0]))
        regime.append(str(x.mask_regime.iloc[0]) if strato == "mascherato" else "")
        ti = _indice_istante(x.timestamp)
        li = x.link_id.map(pos).to_numpy()
        if np.isnan(li.astype(float)).any():
            raise ValueError(f"{f}: tratti fuori dalla rete")
        li = li.astype(np.int64)
        for c in CANALI:
            out[c][d, ti, li] = x[c].to_numpy(np.float32)
        pct[d, ti, li] = x.pct_observed.to_numpy(np.uint8)
        ele[d, ti, li] = x.is_score_eligible.to_numpy().astype(bool)
    ordine_date = np.argsort(np.array(date))
    t = Tensori(
        pannello, split, strato,
        date=np.array(date)[ordine_date], regime=np.array(regime)[ordine_date], link_id=ordine.astype(str),
        speed=out["speed_kmh"][ordine_date], flow=out["flow_vph"][ordine_date],
        occ=out["occupancy"][ordine_date], dens_occ=out["density_occ_linear_vehpkm"][ordine_date],
        pct=pct[ordine_date], eleggibile=ele[ordine_date],
    )
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(percorso, **{k: getattr(t, k) for k in
                              ("date", "regime", "link_id", "speed", "flow", "occ", "dens_occ", "pct", "eleggibile")})
    return t


def bersagli_task1(pannello: str, split: str, t: Tensori) -> pd.DataFrame:
    """Le righe del Task 1 come le chiede il modello, con gli indici (giorno, istante, tratto).

    Il modello è l'elenco esatto delle celle valutate: si legge da lì, non dai NaN.
    """
    m = pd.read_csv(
        RILASCIO / "task1" / pannello / split / "sample_submission_state.csv",
        usecols=["panel", "timestamp", "station_id", "link_id", "mask_regime"], dtype=str,
    )
    gi = {d: i for i, d in enumerate(t.date)}
    li = {l: i for i, l in enumerate(t.link_id)}
    m["g"] = m.timestamp.str.slice(0, 10).map(gi).astype(np.int64)
    m["t"] = _indice_istante(m.timestamp)
    m["l"] = m.link_id.map(li).astype(np.int64)
    return m


def validita_rampe(pannello: str, split: str, cache: bool = True) -> np.ndarray:
    """(D, 288, L) bool: la transizione che parte dalla cella ha le rampe richieste dal valutatore del Task 3.

    Come derive_physics: un tratto senza rampe attaccate (topologia) è valido per struttura;
    uno con rampe d'ingresso o d'uscita attaccate è valido solo se in quella cella c'è almeno
    un'osservazione di rampa di quel tipo eleggibile, con pct_observed ≥ 75 e flusso finito
    (le rampe si collegano ai tratti con ramp_attachment_map). Servono entrambi i tipi.
    """
    percorso = CACHE / f"{pannello}__{split}__rampe.npz"
    if cache and percorso.exists():
        return np.load(percorso)["valida"]
    tr = tratti(pannello)
    li = {l: i for i, l in enumerate(tr.link_id)}
    rete = RILASCIO / "corridors" / pannello / "network"
    topo = pd.read_csv(rete / "lwr_mainline_topology.csv", dtype=str)
    attaccati = {k: np.zeros(len(tr), bool) for k in ("OR", "FR")}
    for k, col in (("OR", "on_ramp_link_ids"), ("FR", "off_ramp_link_ids")):
        for l in topo.loc[topo[col].fillna("").str.len() > 0, "link_id"]:
            if l in li:
                attaccati[k][li[l]] = True
    amap = pd.read_csv(rete / "ramp_attachment_map.csv", dtype=str).set_index("ramp_link_id").nearest_mainline_link_id
    giorni = sorted((RILASCIO / "corridors" / pannello / split / "ramp_states").glob("**/*.parquet"))
    date = tensori(pannello, split).date
    gi = {d: i for i, d in enumerate(date)}
    oss = {k: np.zeros((len(date), ISTANTI, len(tr)), bool) for k in ("OR", "FR")}
    for f in giorni:
        d = pq.read_table(f, columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "pct_observed", "is_score_eligible"]).to_pandas()
        d = d[d.is_score_eligible.astype(bool) & (d.pct_observed >= 75) & d.flow_vph.notna()]
        l = d.ramp_link_id.astype(str).map(amap).map(li)
        g = d.timestamp.astype(str).str.slice(0, 10).map(gi)
        ok = l.notna() & g.notna()
        d, l, g = d[ok], l[ok].astype(int).to_numpy(), g[ok].astype(int).to_numpy()
        t = _indice_istante(d.timestamp)
        for k in ("OR", "FR"):
            m = (d.ramp_type.str.upper() == k).to_numpy()
            oss[k][g[m], t[m], l[m]] = True
    valida = (~attaccati["OR"] | oss["OR"]) & (~attaccati["FR"] | oss["FR"])
    if cache:
        np.savez_compressed(percorso, valida=valida)
    return valida
