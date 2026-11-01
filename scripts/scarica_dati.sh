#!/usr/bin/env bash
# Scarica ed estrae i dati di TrafficFlowBench in data/ (~13 GB estratti).
# Prima: iscrizione alla competizione con regole accettate e credenziali API Kaggle configurate.
# La radice che gli script ufficiali chiamano --release-root è data/kaggle_public.
set -euo pipefail

cd "$(dirname "$0")/.."
C=2026-ieee-big-data-traffic-flow-bench
mkdir -p data
uv run kaggle competitions download -c "$C" -p data
unzip -o -q "data/$C.zip" -d data
rm "data/$C.zip"
ls -la data/kaggle_public
