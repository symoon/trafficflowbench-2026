"""Task 2: finestre del train ricostruite con la regola di selezione pubblicata (outputs/task2/finestre_train.parquet).

Sono le finestre «attuali» su cui si allenano onset e ongoing (allena_task2_finale.py, allena_task2_stack.py).
Il file del 18/10/2026 era stato scritto a mano con queste stesse righe; rigenerato il 31/10 coincide
esattamente (4.196 finestre).

    uv run python scripts/finestre_train.py
"""
import pandas as pd

from cv_task2 import PANNELLI, USCITA
from tfb.finestre import costruisci


def main() -> None:
    USCITA.mkdir(parents=True, exist_ok=True)
    w = pd.concat([costruisci(p) for p in PANNELLI], ignore_index=True)
    w.to_parquet(USCITA / "finestre_train.parquet", index=False)
    print(f"{len(w):,} finestre: {w.condition.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
