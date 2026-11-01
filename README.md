# TrafficFlowBench 2026 – Team Simone Zappatini – reproduction package

2026 IEEE Big Data Cup – TrafficFlowBench (Kaggle `2026-ieee-big-data-traffic-flow-bench`).
Team: Simone Zappatini (Kaggle `simonezappatini`), Independent, szappatini@gmail.com.

| Selected final submission | Public LB | sha256 of the submitted CSV |
|---|---:|---|
| **v19** (primary) | 0.86106 | `f167c63ab25d4e90fab1d3049512c4c09472889cf78c416504ec092154934ef9` |
| v18 | 0.85998 | `f8e8669675b2546951b9952441745b2ec720c72f9ee9c5748017c8058199ee0b` |

v18 and v19 share every step except the main Task 1 model (v19 adds the E5 low-rank month features).
The method, the validation and the analysis are in [`report/TrafficFlowBench_SimoneZappatini_report.pdf`](report/TrafficFlowBench_SimoneZappatini_report.pdf).

## Requirements

- Linux x86-64 (tested on Ubuntu 26.04, kernel 7.0), [uv](https://docs.astral.sh/uv/) ≥ 0.9 (it installs Python 3.12.8 and the pinned dependencies from `uv.lock`).
- CPU only, no GPU needed. Times below are on an Intel i9-11900 (16 threads) with 61 GB RAM; the scripts use the thread counts of the original runs (16), fewer cores work, just slower.
- RAM: 32 GB recommended. The peak is 25 GB in the Task 2 stack step (20 GB in the Task 2 LightGBM step, at most 11 GB elsewhere). Inference only: 5 GB.
- Disk: about 12 GB for the extracted competition data, plus about 17 GB for the Python environment (6 GB), the data cache (5 GB) and the intermediate features and models (6.5 GB).
- Runtime on the machine above: 6 h 31 min for v19 from scratch (3 h 04 min of it for the E2 network on CPU, 1 h 44 min for the Task 2 stack); v18 afterwards adds 23 min, because it reuses every step except the main Task 1 model. Inference only, with the frozen models: about 21 min including the data cache build.
- Kaggle API credentials (`~/.kaggle/kaggle.json`) only to download the data.

## Reproduce

```bash
scripts/scarica_dati.sh          # downloads and extracts the competition data into data/kaggle_public (~12 GB)
scripts/riproduci.sh v19         # full run: environment, features, training, inference
sha256sum submissions/v19.csv    # compare with riferimento/v19.sha256
scripts/riproduci.sh v18         # second selected submission (reuses the shared steps)
```

`scripts/riproduci.sh` is the single entry point. It runs `uv sync --frozen` and then each step once; completed
steps are skipped on a rerun (markers in `outputs/riproduzione/`, together with the log, the wall time and the peak
memory of every step). Delete `outputs/` and `data/cache/` to start from scratch.

Inference only, with the models we trained: download `frozen_models.zip` from the release
<https://github.com/symoon/trafficflowbench-2026/releases/tag/final> (the zip package attached to the same release
already contains it) and unzip it in the repository root. Each folder holds the models and the manifest with their
sha256 hashes written when the submission was generated.

```bash
unzip frozen_models.zip                                        # -> frozen_models/v18, frozen_models/v19
scripts/riproduci.sh v19 --frozen-models frozen_models/v19     # about 21 min, peak 5 GB
```

To compare a regenerated file with the submitted one, download the files we uploaded to Kaggle
(`submitted_v19.zip`, `submitted_v18.zip`) from the same release. `confronta_invio.py` checks the exact hash and,
if the files differ, reports the per-task differences and upper bounds on the change of the private score due to
S_state and S_queue. `confronta_fisica.py` adds the physics term from the saved states of two runs.

```bash
mkdir -p submitted && unzip -d submitted submitted_v19.zip                 # -> submitted/v19.csv
.venv/bin/python scripts/confronta_invio.py submissions/v19.csv submitted/v19.csv
.venv/bin/python scripts/confronta_fisica.py outputs/invii/v19 <folder with the outputs/invii/v19 of another run>
```

### Verification we ran (31 October – 1 November 2026)

On a fresh clone of this repository (new environment from `uv.lock`, data cache rebuilt from the raw release,
empty `outputs/`), on the machine described above:

1. **Inference with the frozen models.** `submissions/v19.csv` is **byte-identical** to the file uploaded to Kaggle
   (sha256 in `riferimento/`). For `submissions/v18.csv`, Task 2 and Task 4 are identical and Task 1 is identical up
   to the last bit of float32: 170 of 6.7 million cells differ by 0.001 in the last decimal written to the CSV, and
   the score change is below 10⁻⁶. The Task 1 part of v18 was computed on 20 October for submission v14, with the
   same models, and reused unchanged up to v18, because v15–v18 changed only Task 2. Recomputed today, it differs
   only in floating-point rounding, so the v18 hash in `riferimento/` is not matched exactly.
2. **Full run, training included** (`scripts/riproduci.sh v19`, then `v18`). The files are not byte-identical to
   the submitted ones, for two reasons. Multi-threaded LightGBM training is not bit-deterministic across runs: the
   first trees of each model coincide, later ones differ in the last digits of thresholds and leaf values. And the
   E2 network was originally trained on a GPU, while here it is trained on the CPU. The XGBoost model and the Task 2
   onset model come out identical. Differences measured with `scripts/confronta_invio.py` and
   `scripts/confronta_fisica.py`:

| Retrained vs submitted | v19 | v18 |
|---|---|---|
| Task 1 cells differing by more than 0.001 | 2.6% (176,930 of 6,735,795) | 2.6% (176,931) |
| Task 1 RMSE of the difference, private | 0.10 km/h, 0.96 veh/h per lane | 0.10 km/h, 0.96 veh/h per lane |
| Task 2 cells differing | 60 of 174,000, in 32 of 160 windows (mean IoU between the files 0.9905) | same |
| Task 4 | identical | identical |
| Upper bound on the change of the private total: S_state | 0.00036 | 0.00036 |
| Upper bound on the change of the private total: S_queue | 0.00277 | 0.00277 |
| Estimated change of the private total: S_physics | 0.00006 | 0.00006 |
| **Total** | **≤ 0.0032** | **≤ 0.0032** |

The two bounds are exact, without the ground truth. S_state is 1 − RMSE/scale per regime and RMSE obeys the
triangle inequality, so the RMSE against any truth changes at most by the RMSE between the two files. For S_queue,
1 − IoU is a metric (Jaccard distance), so the IoU of a window changes at most by 1 − IoU(file A, file B). The
physics term is exact for S_FD and estimated for S_LWR, whose normalisation needs the truth. The total change is at
most 0.0032, about 0.4% of the score, within the 1% tolerance (≈ 0.0086).

## Steps of the full run

| Step | Script | What it does | Wall time | Peak RAM |
|---|---|---|---:|---:|
| `finestre_train` | `finestre_train.py` | Task 2 training windows from the train month (published selection rule) | 1 min 29 s | 1.3 GB |
| `prepara_task1` | `prepara_task1.py` | Task 1 features on the train cells | 3 min 47 s | 6.1 GB |
| `prepara_task1_mondi` | `prepara_task1_mondi.py --copie 1` | the same features in shifted worlds | 5 min 31 s | 5.0 GB |
| `mese_prepara` | `cv_mese.py prepara` | month-level features | 34 s | 2.1 GB |
| `buchi_prepara` | `cv_buchi.py prepara` | Task 2 holes reproduced in train | 6 min 18 s | 6.1 GB |
| `buchi_copie` | `cv_buchi.py copie --copie 3` | three extra copies of the holes | 13 min 24 s | 6.3 GB |
| `finale_buchi` | `allena_task1_finale.py --buchi --copie 3` | LightGBM model for the cells around the Task 2 windows | 23 min 42 s | 10.7 GB |
| `e2_prepara` | `e2_sequenze.py prepara` | input sequences of the E2 network | 6 min 34 s | 5.0 GB |
| `e2_finale` | `e2_sequenze.py finale` | E2 convolutional network, 30 epochs on CPU | 3 h 03 min 50 s | 5.9 GB |
| `e5_prepara` (v19) | `e5_basso_rango.py prepara` | E5 low-rank month completion | 54 s | 1.5 GB |
| `e5_prepara_mondi` (v19) | `e5_basso_rango.py prepara_mondi` | E5 in the shifted worlds | 1 min 21 s | 2.4 GB |
| `finale_e5` (v19) | `allena_task1_finale.py --mondi --mese --e5` | main Task 1 LightGBM model | 9 min 44 s | 7.9 GB |
| `finale_mese` (v18) | `allena_task1_finale.py --mondi --mese` | main Task 1 LightGBM model without E5 | 9 min 51 s | 8.3 GB |
| `task2_finale` | `allena_task2_finale.py --semi 6` | Task 2 LightGBM onset and ongoing models, 6 seeds | 15 min 31 s | 19.6 GB |
| `task2_stack_ongoing` | `allena_task2_stack.py --condizione ongoing` | ongoing stack: LightGBM, XGBoost, CatBoost | 1 h 43 min 51 s | 25.0 GB |
| `genera_v19` / `genera_v18` | `genera_invio.py` | inference, submission file, manifest with hashes | 14 min 06 s / 13 min 24 s | 4.7 / 4.3 GB |

All seeds are fixed (LightGBM, XGBoost and CatBoost seed 0; PyTorch seed 0; NumPy generators seeded per corridor and
block). Thread counts are those of the original runs (16; 15 for the Task 2 stack). Multi-threaded gradient boosting
can differ in the last digits across machines; the effect is orders of magnitude below the 1% tolerance (see the
verification above).

## Task 2: data up to the forecast origin only

For a window with origin T the forecast uses only `window_history.parquet` (T−60 … T−5), the row at T of the masked
layer of the split, and quantities computed on the training split (historical queue frequency, models, thresholds).
`scripts/verifica_task2_online.py` repeats the organizers' check: it truncates the masked layer and the window
history at T for 48 windows (8 corridors × validation/private × onset/ongoing) and verifies that the forecasts do not
change:

```bash
.venv/bin/python scripts/verifica_task2_online.py modello_e1_onset_stack_ongoing outputs/invii/v19/modelli
# -> "48 finestre provate, 0 con previsioni diverse" (48 windows tested, 0 with different forecasts)
```

## Repository layout

| Path | Content |
|---|---|
| `scripts/riproduci.sh` | single entry point |
| `scripts/genera_invio.py` | inference for a version, submission file and manifest with hashes (`outputs/invii/<version>/manifesto.json`) |
| `scripts/allena_*.py`, `prepara_*.py`, `cv_buchi.py`, `cv_mese.py`, `e5_basso_rango.py`, `e2_sequenze.py`, `finestre_train.py` | feature extraction and training (called by the entry script) |
| `scripts/cv_*.py`, `rimaschera_valutati.py`, … | validation experiments described in the report (not needed to reproduce) |
| `src/tfb/` | method: data loading (`dati.py`), Task 1 (`task1.py`, `caratteristiche.py`, `rete.py`), Task 2 (`task2.py`, `task2_modello.py`, `finestre.py`), Task 4 (`task4.py`), metric estimates (`metrica.py`), submission writer (`invio.py`) |
| `scripts/confronta_invio.py`, `confronta_fisica.py` | comparison of two submission files and bounds on the score change |
| `riferimento/` | sha256 of the submitted CSV files |
| `docs/conformita.md` | compliance notes (Italian) |
| `docs/lab_notebook_it.md` | full lab notebook with every experiment and decision (Italian) |
| `report/` | PDF report and its sources |
| `OSSERVAZIONI.md`, `REVISIONE_AVVERSARIALE*.md` | critical reviews of the approach written during development (Italian) |
| `logs/` | logs of the original training and generation runs |

Links in the Italian notes to `docs/kaggle/` and `docs/forum/` pointed to our local copies of the competition pages and
forum threads; they are not redistributed here (they are other people's texts).

The code and the notebook are written in Italian. Short glossary: *allena* = train, *prepara* = prepare, *genera
invio* = generate submission, *riproduci* = reproduce, *caratteristiche* = features, *buchi* = holes (the corridor
blanking around Task 2 windows), *mondi spostati* = shifted worlds, *mese* = month, *finestre* = windows, *code* =
queues, *tratto* = link, *corridoio* = corridor, *verifica* = check, *modelli congelati* = frozen models.

## External data, models and licences

No external data, no pre-trained models, no paid APIs in the method. Libraries (all OSI-approved licences allowing
commercial use): LightGBM 4.7.0 (MIT), XGBoost 3.4.1 (Apache-2.0), CatBoost 1.2.10 (Apache-2.0), PyTorch 2.14.1
(BSD-3), numpy 2.5.4, pandas 3.0.6, scipy 1.18.1 (BSD-3), pyarrow 26.0.0 (Apache-2.0), matplotlib 3.11.2 (PSF),
kaggle 2.2.4 (Apache-2.0, download only). Development was assisted by an AI programming assistant (Claude Code);
it is not part of the method.
