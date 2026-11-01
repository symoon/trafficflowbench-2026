"""Task 1: allena il modello finale su tutte le celle estratte del train (i tre blocchi).

Il numero di alberi viene dalla validazione a blocchi (cv_task1.py). Salva i due modelli
(residuo di velocità e di flusso per corsia) in outputs/task1/finale_{v,q}.txt.

Con --mondi alle celle del train si aggiungono le stesse celle in mondi spostati
(prepara_task1_mondi.py): è la randomizzazione del mondo validata da cv_mondo.py --allenamento entrambi.

Con --buchi si aggiungono le righe dei bersagli attorno alle finestre del Task 2 calcolate sullo
strato con i buchi che gli organizzatori aprono nei mesi valutati (cv_buchi.py prepara).

Con --copie N si aggiungono anche le copie 1 … N dei buchi (cv_buchi.py copie), con altre finestre.

Con --mese si aggiungono le caratteristiche del livello del mese (cv_mese.py prepara), anche alle
righe dei mondi spostati; non ancora per le righe con i buchi.

Con --e5 si aggiungono le stime a basso rango del mese (e5_basso_rango.py prepara e prepara_mondi), come --mese.

Con --nome i modelli si salvano in outputs/task1/<nome>_{v,q}.txt: il modello sui buchi va in
finale_buchi e lo usa task1.ricostruisci_lgb_buchi solo attorno alle finestre del Task 2.

    uv run python scripts/allena_task1_finale.py --alberi 800 [--thread 16] [--mondi] [--buchi [--copie 3]] [--mese] [--nome finale_buchi]
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb.caratteristiche import base
from tfb.dati import RADICE_PROGETTO, pannelli

CARTELLA = RADICE_PROGETTO / "outputs" / "task1"
META = ["y_v", "y_q", "g", "t", "l", "blocco", "completo", "mask_regime", "timestamp", "station_id", "link_id", "pannello"]
PARAMETRI = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 verbose=-1, seed=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alberi", type=int, required=True)
    ap.add_argument("--thread", type=int, default=8)
    ap.add_argument("--mondi", action="store_true")
    ap.add_argument("--buchi", action="store_true")
    ap.add_argument("--copie", type=int, default=0)
    ap.add_argument("--mese", action="store_true")
    ap.add_argument("--e5", action="store_true", help="stime a basso rango del mese (e5_basso_rango.py prepara e prepara_mondi)")
    ap.add_argument("--nome", default="finale")
    a = ap.parse_args()
    if a.e5 and a.buchi:
        raise SystemExit("--e5 con --buchi: le righe con i buchi non hanno le caratteristiche E5")
    if a.mese and a.buchi:
        raise SystemExit("--mese con --buchi: le righe con i buchi non hanno ancora le caratteristiche del mese")
    if a.copie and not a.buchi:
        raise SystemExit("--copie richiede --buchi")
    t0 = time.time()

    def leggi(cartella: str, suffisso: str = "") -> pd.DataFrame:
        x = pd.concat([pd.read_parquet(CARTELLA / cartella / f"{p}{suffisso}.parquet") for p in pannelli()], ignore_index=True)
        if a.mese and cartella in ("caratteristiche", "caratteristiche_mondi"):
            fine = "" if cartella == "caratteristiche" else "__mondi"
            x = pd.concat([x, pd.concat([pd.read_parquet(CARTELLA / "caratteristiche_mese" / f"{p}{fine}.parquet")
                                         for p in pannelli()], ignore_index=True)], axis=1)
        if a.e5 and cartella in ("caratteristiche", "caratteristiche_mondi"):
            fine = "" if cartella == "caratteristiche" else "__mondi"
            x = pd.concat([x, pd.concat([pd.read_parquet(CARTELLA / "caratteristiche_e5" / f"{p}{fine}.parquet")
                                         for p in pannelli()], ignore_index=True)], axis=1)
        return x

    d = leggi("caratteristiche")
    colonne = [c for c in d.columns if c not in META]
    if a.mondi:
        d = pd.concat([d, leggi("caratteristiche_mondi").drop(columns="copia")], ignore_index=True)
    if a.buchi:
        b = leggi("caratteristiche_buchi")
        pezzi = [d, b.loc[b.strato == "buchi", list(d.columns)]]
        pezzi += [leggi("caratteristiche_buchi", f"__copia{c}")[list(d.columns)] for c in range(1, a.copie + 1)]
        d = pd.concat(pezzi, ignore_index=True)
    for nome, y in (("v", d.y_v.to_numpy()), ("q", d.y_q.to_numpy())):
        r = y - base(d, nome)
        ok = np.isfinite(r)
        m = lgb.train(dict(PARAMETRI, num_threads=a.thread), lgb.Dataset(d.loc[ok, colonne], r[ok]), num_boost_round=a.alberi)
        m.save_model(str(CARTELLA / f"{a.nome}_{nome}.txt"))
        print(f"modello {nome}: {ok.sum():,} righe ({time.time() - t0:.0f} s)", flush=True)
    (CARTELLA / f"{a.nome}_parametri.json").write_text(json.dumps(dict(PARAMETRI, num_threads=a.thread, alberi=a.alberi, mondi=a.mondi, buchi=a.buchi, copie=a.copie, mese=a.mese, e5=a.e5, colonne=colonne), indent=1))


if __name__ == "__main__":
    main()
