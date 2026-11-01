"""Script d'ingresso unico: rigenera da data/kaggle_public l'invio di una versione.

    uv run python scripts/genera_invio.py v1 [--riusa]

Le previsioni per compito si salvano in outputs/invii/<versione>/; con --riusa si
ripartisce da lì. Il file da caricare su Kaggle è submissions/<versione>.zip.

Ogni versione congela i suoi modelli in outputs/invii/<versione>/modelli/ e scrive un
manifesto con gli hash: se la cartella esiste, la versione si rigenera da lì e non dai
modelli correnti, che un nuovo allenamento può sovrascrivere. Con --riusa le previsioni
salvate si accettano solo se coincidono con quelle del manifesto.
"""
import argparse
import hashlib
import json
import platform
import shutil
import time
from importlib.metadata import version

import numpy as np
import pandas as pd

from tfb import dati, task1, task2, task4
from tfb.dati import RADICE_PROGETTO, pannelli, tensori
from tfb.invio import SPLIT_VALUTATI, scrivi

VERSIONI = {
    "v1": dict(stato=task1.ricostruisci_v1, code=task2.persistenza, odme=task4.nnls_prior),
    # cambia solo il Task 2 rispetto a v1
    "v3": dict(stato=task1.ricostruisci_v1, code=task2.modello, odme=task4.nnls_prior),
    # cambia solo il Task 4 rispetto a v3: S_od non si misura in locale
    "v4": dict(stato=task1.ricostruisci_v1, code=task2.modello, odme=task4.kl_prior),
    # cambia solo il Task 4 rispetto a v3: prior medio dei tre split
    "v5": dict(stato=task1.ricostruisci_v1, code=task2.modello, odme=task4.nnls_prior_medio),
    # cambia solo il Task 1 rispetto a v3: Δ = 0,35·ΔS_state (noto in locale) + 0,15·ΔS_physics
    "v6": dict(stato=task1.ricostruisci_lgb, code=task2.modello, odme=task4.nnls_prior),
    # come v6 con il Task 1 a 3200 alberi (CV a tre fold: S_state 0,9393 -> 0,9411, tutti i corridoi)
    "v7": dict(stato=task1.ricostruisci_lgb, code=task2.modello, odme=task4.nnls_prior),
    # come v6, ongoing allenato su finestre attuali + casuali (0,857 contro 0,835 sulle ongoing ufficiali fuori campione)
    "v8": dict(stato=task1.ricostruisci_lgb, code=task2.modello, odme=task4.nnls_prior),
    # come v8, Task 1 allenato sul train più una copia in mondi spostati (cv_mondo: S_state nei mondi spostati 0,9346 -> 0,9367)
    "v9": dict(stato=task1.ricostruisci_lgb, code=task2.modello, odme=task4.nnls_prior),
    # come v8, Task 1 con un secondo modello attorno alle finestre del Task 2, allenato sui loro buchi riprodotti nel train
    "v10": dict(stato=task1.ricostruisci_lgb_buchi, code=task2.modello, odme=task4.nnls_prior),
    # come v10 con E1: lo scarto del livello notturno del mese da quello del train si toglie dagli ingressi del Task 1
    # e si restituisce alle velocità previste; stessi modelli (rimascheratura dei mesi veri: S_state +0,0027/+0,0029)
    "v11": dict(stato=task1.ricostruisci_lgb_buchi_e1, code=task2.modello, odme=task4.nnls_prior),
    # come v11, modello sui buchi allenato anche sulle copie 1–3 dei buchi simulati (pseudo-mesi: +0,00077, 69/3)
    "v12": dict(stato=task1.ricostruisci_lgb_buchi_e1, code=task2.modello, odme=task4.nnls_prior),
    # come v12, modello principale con il livello del mese, allenato su train + mondi spostati (pseudo-mesi: +0,0006 su v12)
    "v13": dict(stato=task1.ricostruisci_lgb_buchi_e1, code=task2.modello, odme=task4.nnls_prior),
    # come v13, nella regione delle finestre media del modello sui buchi e della rete E2 (pseudo-mesi: +0,00047, 63/9);
    # serve PyTorch (gruppo gpu): il Task 1 si calcola con scripts/stato_server.py v14 ricostruisci_lgb_buchi_rete_e1
    "v14": dict(stato=task1.ricostruisci_lgb_buchi_rete_e1, code=task2.modello, odme=task4.nnls_prior),
    # come v14, ongoing allenato su attuali + casuali dei semi 0–5 (allena_task2_finale.py --semi 6; ongoing casuali
    # fuori campione 0,8294 -> 0,8423, 3 fold su 3; ufficiali 0,8527 -> 0,8586); Task 1 e Task 4 identici a v14
    "v15": dict(stato=task1.ricostruisci_lgb_buchi_rete_e1, code=task2.modello, odme=task4.nnls_prior),
    # come v15, onset a T+30 dallo stack di LightGBM, XGBoost e CatBoost con il contesto della finestra
    # (allena_task2_stack.py; cv_task2_stack.py, onset attuali 0,8649 -> 0,8787, 3 fold su 3); ongoing, Task 1 e 4 come v15
    "v16": dict(stato=task1.ricostruisci_lgb_buchi_rete_e1, code=task2.modello_stack, odme=task4.nnls_prior),
    # come v15, onset del Task 2 con E1: velocità di storia e di T riportate nel mondo del train con lo scarto del
    # livello notturno del mese (notti fino a T), stessi modelli (cv_task2_stack.py mondo --e1: mondo del train −0,0007,
    # mondi spostati +0,0120 e +0,0067); ongoing, Task 1 e 4 come v15
    "v17": dict(stato=task1.ricostruisci_lgb_buchi_rete_e1, code=task2.modello_e1_onset, odme=task4.nnls_prior),
    # come v17, ongoing dallo stack di LightGBM, XGBoost e CatBoost con il contesto della finestra (allena_task2_stack.py
    # --condizione ongoing; ongoing casuali 0,8423 -> 0,8650, 3 fold su 3); onset, Task 1 e 4 come v17
    "v18": dict(stato=task1.ricostruisci_lgb_buchi_rete_e1, code=task2.modello_e1_onset_stack_ongoing, odme=task4.nnls_prior),
    # come v18, modello principale del Task 1 con E5: stime a basso rango del mese tra le caratteristiche
    # (allena_task1_finale.py --mondi --mese --e5; CV a tre blocchi sopra il ramo del mese +0,00105, 25/30); Task 2 e 4 come v18
    "v19": dict(stato=task1.ricostruisci_lgb_buchi_rete_e1, code=task2.modello_e1_onset_stack_ongoing, odme=task4.nnls_prior),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("versione", choices=sorted(VERSIONI))
    ap.add_argument("--riusa", action="store_true", help="riusa le previsioni già salvate per compito")
    a = ap.parse_args()
    cfg = VERSIONI[a.versione]
    cartella = RADICE_PROGETTO / "outputs" / "invii" / a.versione
    cartella.mkdir(parents=True, exist_ok=True)
    congelati = cartella / "modelli"
    if congelati.exists():
        dati.MODELLI = congelati
    if a.riusa:
        _verifica_riuso(cartella)
    t0 = time.time()
    stato = {}
    for p in pannelli():
        for s in SPLIT_VALUTATI:
            f = cartella / f"stato_{p}_{s}.npz"
            if a.riusa and f.exists():
                z = np.load(f)
                stato[(p, s)] = (z["v"], z["q"])
            else:
                stato[(p, s)] = cfg["stato"](tensori(p, s))
                np.savez(f, v=stato[(p, s)][0], q=stato[(p, s)][1])
        print(f"Task 1 {p}: fatto ({time.time() - t0:.0f} s)", flush=True)
    tabelle = {}
    for nome in ("code", "odme"):
        f = cartella / f"{nome}.parquet"
        if a.riusa and f.exists():
            tabelle[nome] = pd.read_parquet(f)
        else:
            tabelle[nome] = pd.concat([cfg[nome](p, s) for p in pannelli() for s in SPLIT_VALUTATI], ignore_index=True)
            tabelle[nome].to_parquet(f, index=False)
        print(f"{nome}: {len(tabelle[nome]):,} righe ({time.time() - t0:.0f} s)", flush=True)
    zp = scrivi(stato, tabelle["code"], tabelle["odme"], RADICE_PROGETTO / "submissions" / f"{a.versione}.csv")
    manifesto(a.versione, cartella, zp)
    print(f"Scritto {zp} ({zp.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f} s")


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blocco in iter(lambda: f.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


INTERMEDI = ("stato_*.npz", "code.parquet", "odme.parquet")


def _verifica_riuso(cartella) -> None:
    """Le previsioni salvate valgono solo se sono quelle registrate nel manifesto della versione."""
    f = cartella / "manifesto.json"
    if not f.exists():
        return  # versione nuova: gli intermedi copiati a mano finiscono nel manifesto alla fine
    attesi = json.loads(f.read_text()).get("intermedi", {})
    for nome, h in attesi.items():
        p = cartella / nome
        if p.exists() and _sha(p) != h:
            raise SystemExit(f"{p}: diverso da quello del manifesto; togliere --riusa o il file")


def manifesto(versione: str, cartella, zp) -> None:
    """Congela i modelli usati e collega l'invio ai suoi file: modelli, intermedi, rilascio, versioni."""
    from tfb.dati import RILASCIO
    cfg = VERSIONI[versione]
    usa = {"task1": cfg["stato"].__name__ != "ricostruisci_v1", "task2": cfg["code"].__name__.startswith("modello")}
    modelli = sorted(p for d, serve in usa.items() if serve for pat in ("finale_*", "rete_*")
                     for p in (dati.MODELLI / d).glob(pat))
    congelati = cartella / "modelli"
    if dati.MODELLI != congelati:
        for p in modelli:
            (congelati / p.parent.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, congelati / p.parent.name / p.name)
    m = dict(
        versione=versione,
        metodi={k: f"{f.__module__}.{f.__name__}" for k, f in cfg.items()},
        modelli={f"{p.parent.name}/{p.name}": _sha(p) for p in modelli},
        intermedi={p.name: _sha(p) for pat in INTERMEDI for p in sorted(cartella.glob(pat))},
        rilascio={str(p.relative_to(RADICE_PROGETTO)): _sha(p) for p in (RILASCIO / "config" / "corridors.json",
                                                                          RILASCIO / "sample_submission.csv", RILASCIO / "submission_key.csv")},
        invio={zp.name: _sha(zp)},
        python=platform.python_version(),
        librerie={n: version(n) for n in ("lightgbm", "numpy", "pandas", "scipy", "pyarrow")
                  + (("xgboost-cpu", "catboost") if "stack" in cfg["code"].__name__ else ())},
    )
    (cartella / "manifesto.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
