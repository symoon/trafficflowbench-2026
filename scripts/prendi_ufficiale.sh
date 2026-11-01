#!/usr/bin/env bash
# Clona in ufficiale/ il repository degli organizzatori (valutatori, baseline, schemi) al commit fissato.
# Per seguire un aggiornamento: cambiare COMMIT e rilanciare.
set -euo pipefail

cd "$(dirname "$0")/.."
URL=https://github.com/jacky850/trafficflowbench-public.git
COMMIT=f9e3fbc  # 14/10/2026, "Say that the row at T may be used, and describe the persistence baseline accurately"

if [ ! -d ufficiale/.git ]; then
    git clone -q "$URL" ufficiale
fi
git -C ufficiale fetch -q origin
git -C ufficiale checkout -q "$COMMIT"
git -C ufficiale log -1 --format='%h %ci %s'
