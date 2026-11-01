# Conformità alle regole (verifica del 19/10/2026)

Fonti: pagine Kaggle riscaricate il 19/10 (identiche alle copie in `docs/kaggle/` del 18/10; nuove solo «Organizers and Judging» e «References», informative) e risposte degli organizzatori in `docs/forum/`.

## Task 2: solo dati con timestamp ≤ T

Regola (Rules, forum 742068, 748789, 749601): per una finestra con origine T nessun dato di traffico successivo a T, né come input né «indirettamente attraverso adattamento, selezione o valori scritti nel metodo». In premiazione rilanciano il codice togliendo ogni riga successiva a T e controllano che le previsioni non cambino.

| Elemento | Da dove viene | Timestamp |
|---|---|---|
| Storia T-60 … T-5 | `window_history.parquet` | < T |
| Riga a T | strato mascherato dello split | = T |
| Frequenza storica di coda per tratto e ora | tutto il train (giu 2030 – feb 2031) | prima di validation e private |
| Modelli LightGBM | finestre ricostruite dal solo train | prima di validation e private |
| Soglie 0,40 / 0,45 | validazione a blocchi sul solo train | prima di validation e private |

Prova: `scripts/verifica_task2_online.py` tronca a T strato mascherato e `window_history` per 48 finestre (8 corridoi, validation e private, onset e ongoing): previsioni identiche in tutte e 48.

Vincolo che ne segue: nessuna scelta del Task 2 si fa guardando la classifica pubblica, perché le sue etichette sono successive a T.

## Task 1, 3, 4: offline

Si può usare ogni osservazione rilasciata dello split, anche successiva alla cella. Task 1 usa lo strato mascherato dello split e profili e modelli dal train. Il Task 4 in uso (NNLS con λ = 0,05) usa conteggi e prior dello split.

Da v10 il Task 1 legge anche `task2/<corridoio>/<split>/window_index.csv`, file rilasciato, solo per sapere dove gli organizzatori hanno oscurato il corridoio attorno alle finestre del Task 2 (MASK_SPEC.md). Lì usa un secondo modello allenato sul solo train con quei buchi riprodotti (`scripts/cv_buchi.py`, `task1.ricostruisci_lgb_buchi`). Il Task 2 non cambia: non usa il Task 1. Le verifiche con rimascheratura dei mesi valutati (`scripts/rimaschera_valutati.py`) usano solo gli ingressi rilasciati per confrontare modelli già allenati; non allenano né scelgono iperparametri.
Da v11 il Task 1 stima anche, per ogni tratto, il livello di velocità notturno del mese dallo strato mascherato dello split (osservazioni rilasciate, consentite per il Task 1 offline) e lo confronta con quello del train (E1, `task1.equivariante`). Il Task 2 non lo usa.
Le varianti v4 (KL) e v5 (media dei prior dei tre split) erano prove, scartate; nessuna entra nell'invio finale.

## Dati, modelli e strumenti esterni

- Dati esterni: nessuno. Modelli preaddestrati: nessuno. API a pagamento nel metodo: nessuna.
- Librerie, tutte con licenza approvata OSI che permette l'uso commerciale: LightGBM 4.7.0 (MIT), numpy 2.5.4 (BSD-3), pandas 3.0.6 (BSD-3), scipy 1.18.1 (BSD-3), pyarrow 26.0.0 (Apache-2.0), matplotlib 3.11.2 (PSF), kaggle 2.2.4 (Apache-2.0, solo per scaricare e inviare). PyTorch 2.14.1 (BSD-3), nel gruppo facoltativo `gpu`: dal 20/10 serve al metodo, perché v14 usa la rete E2 nelle finestre del Task 2 (allenata e usata sulla CPU, nessuna GPU necessaria). Se l'invio scelto è v13 o precedente, non serve. XGBoost (`xgboost-cpu` 3.4.1, Apache-2.0, gruppo facoltativo `xgb`) e CatBoost (`catboost` 1.2.10, Apache-2.0, gruppo facoltativo `catboost`): dal 28/10 servono solo a v16, che li usa nello stack dell'onset del Task 2 (`task2.modello_stack`); v15 e precedenti non li usano. Il repository degli organizzatori è MIT.
- Sviluppo assistito da un assistente di programmazione AI (Claude Code). Non fa parte del metodo e non produce previsioni; va dichiarato nel report per trasparenza, in linea con la politica IEEE sui contenuti generati con l'AI.
- Calcolo su macchine proprie (locale e server 192.168.0.3).

## Squadra, condivisione, invii

- Un solo account (simonezappatini). Nessun codice o dato condiviso fuori dalla squadra.
- Il codice non va pubblicato (GitHub pubblico o altro) prima della fine della gara: le regole generali ammettono la condivisione pubblica solo sul forum o nei notebook Kaggle. Per il pacchetto finale: zip o repository privato condiviso con gli organizzatori.
- Limite di 5 invii al giorno (UTC) rispettato. Tutti gli invii sono successivi alla ricostruzione dei dati dell'11/09/2026, quindi validi per i premi.
- Niente etichettatura a mano, niente ricostruzione delle etichette nascoste: le scelte si fanno modellando i dati rilasciati.

## Da fare per i premi

1. ~~Registrazione via e-mail entro il 25/10/2026 alle 23:59 AoE~~ fatta il 19/10/2026 da szappatini@gmail.com (testo in `docs/registrazione.md`); affiliazione «Independent» confermata con una risposta agli organizzatori lo stesso giorno. I membri devono coincidere con la squadra Kaggle: un solo membro.
2. (Stato al 31/10/2026: script d'ingresso unico `scripts/riproduci.sh`, README in inglese `README_EN.md`, report in `docs/report/`, pacchetto con `scripts/pacchetto.sh`; consegna come repository GitHub `symoon/trafficflowbench-2026`, privato fino alla chiusura degli invii e poi pubblico, scelta dell'utente del 31/10. Verifica del 31/10–01/11 su un clone pulito: dai modelli congelati v19 identico byte per byte, v18 identico salvo l'ultimo bit di float32 in Task 1; riallenando da zero (6 h 31 min) scarto sul punteggio privato al massimo 0,0032, sotto la soglia dell'1%.) Pacchetto entro il 10/11/2026 alle 23:59 AoE: codice con README, ambiente, un solo script d'ingresso che rifà da `data/kaggle_public` l'invio scelto (anche l'allenamento), seed e versioni, dichiarazione di dati esterni e licenze (questo file), tempi e hardware; report PDF.
3. Riproducibilità entro l'1% del punteggio privato: l'allenamento LightGBM a più thread può differire di poco tra macchine; fissare `deterministic` e `seed` e verificare un rilancio da zero, cache comprese (la cache in `data/cache` va ricostruita, non riusata).
