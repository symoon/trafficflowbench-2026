"""Calcola sul server le previsioni del Task 1 di una versione per validation e private (stato_{p}_{s}.npz).

    python scripts/stato_server.py v10 [ricostruisci_lgb_buchi]

Il metodo è una funzione di tfb.task1 (default ricostruisci_lgb). Se esiste
outputs/invii/<versione>/modelli, i modelli si leggono da lì, come in genera_invio.py.
"""
import sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from tfb import dati, task1
from tfb.dati import RADICE_PROGETTO, pannelli, tensori
from tfb.invio import SPLIT_VALUTATI
uscita = RADICE_PROGETTO / "outputs" / "invii" / sys.argv[1]
uscita.mkdir(parents=True, exist_ok=True)
if (uscita / "modelli").exists():
    dati.MODELLI = uscita / "modelli"
metodo = getattr(task1, sys.argv[2] if len(sys.argv) > 2 else "ricostruisci_lgb")
t0 = time.time()
for p in pannelli():
    for s in SPLIT_VALUTATI:
        v, q = metodo(tensori(p, s))
        np.savez(uscita / f"stato_{p}_{s}.npz", v=v, q=q)
    print(p, f"{time.time() - t0:.0f} s", flush=True)
