"""S_FD esatto delle versioni inviate, in validation e private (metrica.s_fd_valutato).

S_FD non chiede la verità: confronta i flussi con il diagramma triangolare su tutte le celle
eleggibili, con le risposte nelle celle bersaglio e i valori rilasciati altrove. Si leggono le
previsioni salvate in outputs/invii/<versione>/stato_*.npz.

    uv run python scripts/fd_valutati.py v3 v6 v7 v9
"""
import sys

import numpy as np
import pandas as pd

from tfb.dati import RADICE_PROGETTO, bersagli_task1, pannelli, tensori
from tfb.invio import SPLIT_VALUTATI
from tfb.metrica import s_fd_valutato

INVII = RADICE_PROGETTO / "outputs" / "invii"


def main() -> None:
    versioni = sys.argv[1:]
    righe = []
    for p in pannelli():
        for s in SPLIT_VALUTATI:
            m = tensori(p, s)
            b = bersagli_task1(p, s, m)
            g, t, l = b.g.to_numpy(), b.t.to_numpy(), b.l.to_numpy()
            for v in versioni:
                z = np.load(INVII / v / f"stato_{p}_{s}.npz")
                r = s_fd_valutato(b, m, z["v"][g, t, l], z["q"][g, t, l])
                righe.append(dict(pannello=p, split=s, versione=v, **{k: r[k] for k in ("R1", "R2", "R3", "S_FD")}))
        print(p, flush=True)
    tab = pd.DataFrame(righe)
    tab.to_csv(RADICE_PROGETTO / "outputs" / "task1" / "fd_valutati.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nS_FD medio sui corridoi\n", tab.groupby(["versione", "split"]).S_FD.mean().unstack().round(5).to_string())
    print("\nPer corridoio, validation\n", tab[tab.split == "validation"].pivot(index="pannello", columns="versione", values="S_FD").round(4).to_string())


if __name__ == "__main__":
    main()
