"""Task 2: ripete la verifica degli organizzatori (forum 749601).

Per alcune finestre di validation e private si tolgono tutte le righe successive all'origine T
(strato mascherato e window_history) e si controlla che le previsioni non cambino.

    uv run python scripts/verifica_task2_online.py [metodo [cartella dei modelli]]
"""
import dataclasses, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import tfb.task2 as T2
from tfb.dati import tensori, _indice_istante, RILASCIO
PANNELLI = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]
orig_tensori, orig_read = T2.tensori, pd.read_parquet
rng = np.random.default_rng(1)
metodo = sys.argv[1] if len(sys.argv) > 1 else "modello"
if len(sys.argv) > 2:  # modelli congelati di una versione, per esempio outputs/invii/v18/modelli
    from tfb import dati
    dati.MODELLI = dati.RADICE_PROGETTO / sys.argv[2]
tot = diverse = 0
for p in PANNELLI:
    for s in ("validation", "private"):
        T2.tensori = orig_tensori
        pieno = getattr(T2, metodo)(p, s)
        m = orig_tensori(p, s)
        w = T2.finestre(p, s)
        for i in rng.choice(len(w), size=3, replace=False):
            x = w.iloc[i]
            g = list(m.date).index(x.date); t = int(_indice_istante(pd.Series([x.forecast_origin]))[0])
            dopo = np.zeros(m.speed.shape, bool); dopo[g, t + 1:] = True; dopo[g + 1:] = True
            mt = dataclasses.replace(m, **{c: np.where(dopo, np.nan, getattr(m, c)) for c in ("speed", "flow", "occ", "dens_occ")},
                                     pct=np.where(dopo, 0, m.pct), eleggibile=m.eleggibile & ~dopo)
            T2.tensori = lambda pannello, split, *a, **k: mt
            # window_history: via le righe successive a T (le finestre dopo T spariscono, la nostra resta)
            T_ = pd.Timestamp(x.forecast_origin)
            def lettura(path, *a, **k):
                d = orig_read(path, *a, **k)
                if "window_history" in str(path):
                    d = d[pd.to_datetime(d.timestamp) <= T_]
                return d
            T2.pd.read_parquet = lettura
            tronco = getattr(T2, metodo)(p, s)
            T2.pd.read_parquet = orig_read
            a_ = pieno[pieno.window_id == x.window_id].reset_index(drop=True)
            b_ = tronco[tronco.window_id == x.window_id].reset_index(drop=True)
            ok = a_[["timestamp", "link_id"]].equals(b_[["timestamp", "link_id"]]) and (a_.queue_pred.values == b_.queue_pred.values).all()
            tot += 1; diverse += int(not ok)
            print(p, s, x.window_id, x.condition, "uguali" if ok else "DIVERSE", f"({int(a_.queue_pred.sum())} celle in coda)", flush=True)
T2.tensori = orig_tensori
print(f"\n{tot} finestre provate, {diverse} con previsioni diverse")
