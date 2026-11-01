# TrafficFlowBench (2026 IEEE Big Data Cup)

Competizione Kaggle su dieci corridoi autostradali direzionali a 5 minuti: ricostruire le celle mancanti, prevedere le code a 30 minuti, rispettare la fisica del deflusso, stimare la domanda origine-destinazione.
Pagina: https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench · Codice ufficiale: https://github.com/jacky850/trafficflowbench-public · Copia delle pagine al 18/10/2026: `docs/kaggle/`

## Regole che contano

| Voce | Valore |
|---|---|
| Registrazione per i premi | e-mail a trafficflowbench@gmail.com entro **25/10/2026**, 23:59 AoE (testo in `docs/registrazione.md`); i membri dichiarati devono coincidere con la squadra Kaggle |
| Fine invii Kaggle | 07/11/2026, 06:55 UTC, 07:55 ora italiana (la pagina dice 6 novembre; vale l'orario Kaggle) |
| Pacchetto finale | codice + report PDF entro **10/11/2026**, 23:59 AoE: README, ambiente, un solo script d'ingresso, seed, versioni, dichiarazione dati esterni |
| Premi | 1.500 / 1.000 / 500 USD (più 500 per squadre K-12) e invito a presentare a IEEE BigData 2026 (Phoenix, 14–17/12) |
| Assegnazione | prime 10 squadre idonee del privato; il codice va riprodotto entro l'1% del punteggio privato; voto finale = 0,85 × privato normalizzato min-max nel gruppo + 0,15 × report |
| Metrica | `0,35·S_state + 0,30·S_queue + 0,15·S_physics + 0,20·S_ODME`, ogni termine in [0, 1]; media per corridoio (due direzioni), poi sui cinque corridoi |
| Classifica pubblica | mese `validation` (marzo 2031); la privata è `private` (aprile), generata a parte con altri incidenti: le due classifiche non coincideranno |
| Consegna | un CSV (`sample_submission.csv`, ~7 milioni di righe, 152 MB) che copre validation e private insieme |
| Squadre | al massimo 4 persone; niente condivisione privata di codice fuori squadra |
| Dati esterni | ammessi se pubblici e gratuiti; i dati sono sintetici, quindi non esiste una fonte esterna con le risposte |
| Task 2 | previsione online: per l'origine `T` solo dati con timestamp ≤ `T`, in qualunque file |

Limite: 5 invii al giorno.

## Principio: niente overfitting

L'obiettivo non è il punteggio pubblico più alto ma un metodo che regga sul mese privato, generato a parte con altri incidenti.

- Ogni scelta (caratteristiche, parametri, soglie, invii finali) si decide sulla validazione locale del train: blocchi temporali, più fold, guardando la dispersione tra fold e tra corridoi e non solo la media.
- La classifica pubblica serve a controllare che il metodo si trasferisca e a tarare ciò che in locale non si misura (il termine di conservazione del Task 3), mai a regolare parametri.
- A parità di punteggio locale vince il metodo più semplice e più stabile.
- Il Task 2 ha 5 finestre per condizione e corridoio: in classifica è rumorosissimo (deviazione standard dell'IoU per finestra 0,375, dichiarata dagli organizzatori). Si valuta su finestre costruite su tutti i 9 mesi del train con la regola di selezione pubblicata.
- Prima di fidarsi del train si confrontano le statistiche osservabili di validation e private (velocità, flussi, frequenza delle code) con quelle del train.

## I quattro compiti

| Task | Peso | Si consegna | Punteggio | In locale |
|---|---:|---|---|---|
| 1 Stato | 0,35 | `speed_kmh`, `flow_vph` sulle celle oscurate (regimi R1/R2/R3: 20/30/50%) | RMSE velocità /25 km/h e flusso per corsia /600 veic/h | sì, sul train |
| 2 Code | 0,30 | `queue_pred` 0/1 da T+5 a T+30 per ogni link, 10 finestre per corridoio | IoU spazio-temporale; coda = velocità ≤ 0,6 × libera | no |
| 3 Fisica | 0,15 | niente: si calcola sulle risposte del Task 1 | diagramma fondamentale triangolare (1/3) e conservazione LWR con le rampe (2/3) | no (il punteggio locale è privo di senso) |
| 4 ODME | 0,20 | `path_flow` per percorso | accordo OD, conteggi sui link, scostamento dal prior, attrazioni | in parte |

`D12_I405_N` e `D12_I405_S` non hanno finestre del Task 2. Un compito con righe mancanti vale 0.

## Valutazione al 18/10/2026

- 511 squadre in classifica pubblica. 1° 0,924 · 10° 0,900 · 50° 0,883 · 100° 0,871 · 200° 0,854.
- Riferimenti: baseline ufficiali 0,553, risposta perfetta 0,9945. Un notebook pubblico arriva a 0,809 (24/09).
- Il gruppo dei premi si forma tra le squadre registrate per e-mail che consegnano codice riproducibile e report: buona parte delle 511 non lo farà, e il report pesa il 15%.
- 28/10/2026: 648 squadre; 1° 0,928 · 10° 0,903 · 50° 0,886 · 100° 0,876 · 200° 0,868. Noi 273° con v14 (0,85689; il 21/10 era 202°). Sono usciti notebook pubblici da 0,865–0,871 (miscele di altri notebook pubblici); il loro Task 1 vale quanto il nostro (S_state 0,9397, conservazione 0,593), il vantaggio sta nel Task 2. Non ne usiamo modelli né pesi (decisione dell'utente, 28/10).
- La verità del Task 2 viene dallo stato sottostante, non dalle osservazioni (SCORING_SPEC.md, Task 2); le nostre etichette locali sono osservate. Sull'onset non conta: a T+30 solo il 2,2% delle celle in coda osservate sta entro 3 km/h dalla soglia.
- Task 2, finestre `queue_onset`: la coda compare solo a T+30 (confermato dagli organizzatori in [questa discussione](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350); è ammesso usarlo). Resta da indovinare quali link si accodano.

## Ambiente

```bash
uv sync --frozen --group gpu --group xgb --group catboost  # Python 3.12 e dipendenze fissate; da v14 serve PyTorch (rete E2), da v16 XGBoost e CatBoost
scripts/prendi_ufficiale.sh  # repository degli organizzatori in ufficiale/, al commit fissato
scripts/scarica_dati.sh      # dopo aver accettato le regole sulla pagina della competizione
```

La radice dei dati per gli script ufficiali (`--release-root`) è `data/kaggle_public`.

Attenzione: `uv run` senza gli stessi `--group` riallinea l'ambiente e disinstalla i gruppi mancanti (PyTorch compreso). Usare `uv run --frozen --group gpu --group xgb --group catboost …` oppure direttamente `.venv/bin/python`; `esito_invio.py` vuole `.venv/bin` nel PATH. Dal 28/10 PyTorch è installato anche in locale (CPU): il Task 1 di v18 rigenerato qui coincide con quello calcolato sul server entro 2·10⁻⁵ km/h.

Il lavoro è numerico su CPU (qui 16 core, 61 GB di RAM). La GPU su 192.168.0.3 (RTX 4090, 24 GB) è disponibile dal 19/10, se un modello la richiede.

## Cartelle

- `data/`: dati della competizione (fuori da git)
- `ufficiale/`: valutatori, baseline e schemi degli organizzatori (fuori da git)
- `docs/kaggle/`: copia delle pagine della competizione
- `notebooks/`: esplorazione
- `scripts/`: utilità da riga di comando
- `src/tfb/`: codice del metodo
- `submissions/`: file generati in locale (fuori da git)

## Piano

1. Esplorazione: topologia (`lwr_mainline_topology.csv`), parametri del diagramma fondamentale per link, rampe, finestre del Task 2. I dati vengono da un modello di deflusso calibrato con parametri pubblicati: prima di tutto va capito quanto le osservazioni del train seguono quel modello.
2. Task 1 e 3 insieme (metà del punteggio): ricostruzione dallo stesso giorno (link a monte e a valle, istanti vicini) con il profilo storico come prior, poi correzione dei flussi perché rispettino la conservazione con le rampe osservate. Si valuta il Task 1 sul train con `score_task1.py`; il Task 3 solo in classifica.
3. Task 4: minimi quadrati non negativi regolarizzati verso il prior (baseline ufficiale, λ = 0,05), poi taratura sul riferimento locale.
4. Task 2: onset con previsione solo a T+30 sui colli di bottiglia appresi dal train; ongoing con propagazione della coda a monte alla velocità d'onda del diagramma fondamentale. Solo dati ≤ T.
5. Riproducibilità fin dal primo giorno: un solo script d'ingresso che produce il CSV da `data/`, seed e versioni fissati. Report PDF tra il 7 e il 10/11.

Calendario: registrazione entro il 25/10; sviluppo fino al 2/11; scelta degli invii finali entro il 6/11; pacchetto e report entro il 10/11.

## Cosa sappiamo dei dati

- Un rilevatore per tratto, 288 istanti al giorno; ordine da monte a valle per milepost (crescente per N ed E, decrescente per S e W). La topologia del file è spezzata in molti tronconi: lo dicono anche gli organizzatori, e il valutatore del Task 3 ordina per milepost.
- Rumore: in flusso libero la velocità ha rumore bianco di circa 0,85 km/h (limite irriducibile); le fluttuazioni del flusso sono reali, correlate al 90% tra tratti vicini.
- **Ogni mese valutato è un mondo diverso dal train** (`scripts/spostamento_split.py`, `scripts/spostamento_cause.py`, 19/10). Un classificatore distingue a stento due mesi consecutivi del train (AUC 0,47, massimo 0,59) ma riconosce validation rispetto a febbraio (0,70) e private rispetto a validation (0,72), su tutti i 10 corridoi. Nessuna caratteristica da sola li separa e le distribuzioni globali coincidono: li separa la velocità per tratto (AUC 0,66). La velocità media in flusso libero per tratto cambia di 0,03 km/h tra mesi del train e di 1,17 km/h verso validation e private (fino a 3–5 km/h); gli scarti dei due mesi valutati correlano 0,5, cioè sono estrazioni indipendenti. Il flusso per corsia per tratto cambia del 2,5% (1,3% nel train), la quota di coda per tratto di 2,5 punti (1,3); occupazione/flusso è identica. La validazione a blocchi sta tutta in un mondo e non vede questo spostamento: è la spiegazione più probabile del mancato trasferimento di v7, perché un modello che riconosce il tratto (v_f, corsie, capacità, lunghezza) impara correzioni del mondo del train.
- Validation e private assomigliano al train: velocità media, quota di celle in coda (3,6–4,4%) e flussi per corsia stanno dentro la variabilità tra i mesi del train (`scripts/confronta_split.py`). Per singolo corridoio la quota di code oscilla di più, per via degli incidenti.
- Task 3: i flussi di confine sono degli organizzatori, proiettati sulle osservazioni vere, quindi S_LWR = 1 − Σ|ΔN̂ − ΔN| / Σ|ΔN| con N = (q/v)·L su tutte le transizioni dei giorni del regime. `tfb.metrica.s_lwr_stima` ne dà una stima locale. Le maschere sono estratte a caso per cella, quindi una cella isolata pesa 2·|N̂ − N|: conta l'errore sulla densità in L1. Ordini di grandezza (D12_I5_N): N ≈ 5 veicoli, |ΔN| mediana 0,13, errore di v1 su N con mediana 0,12.
- Le rampe non spiegano le differenze di flusso tra rilevatori consecutivi (correlazione 0,08 su D12_I5_N e D7_I210_E): sono sintetiche e scollegate dalla linea principale, quindi non servono al Task 1.
- Stima allineata al valutatore (`tfb.metrica.s_fisica_stima`, 19/10): parametri di rete identici a `network_parameters` su tutti i corridoi, S_FD identico all'ufficiale (0,9907 su D12_I405_S R1), transizioni entro lo 0,12%. Con i filtri ufficiali (tratti osservabili, rampe valide, modalità A) S_LWR di v1 cambia al più di 0,002 rispetto alla stima precedente; S_FD vale 0,981–0,992, come la verità.
- Cambio di mondo simulato (`scripts/cv_mondo.py`, 19/10): nel blocco valutato ogni tratto riceve uno scarto di velocità δ ~ N(0; 1,2 km/h) solo in flusso libero, come tra train e mesi valutati, su ingressi e verità. L'interpolazione non cambia (Δ 0); LightGBM a 800 alberi perde 0,0047 di S_state e 0,0067 di S_LWR (stabile sui tre fold), a 3200 perde 0,0063 e 0,0088, e il suo vantaggio su 800 scende da +0,0018 a +0,0002: coerente con v7 in classifica.
- Randomizzazione del mondo in allenamento (`scripts/prepara_task1_mondi.py`, `cv_mondo.py --allenamento`, 19/10): le stesse celle del train ricalcolate in un mondo spostato per blocco (δ ~ N(0; 1,2), semi indipendenti da quelli di valutazione). Allenando su train + mondi spostati LightGBM a 800 alberi passa nei mondi spostati da 0,9346 a 0,9367 di S_state e da 0,5898 a 0,5948 di S_LWR, e nel mondo del train perde 0,0004 (0,9393 → 0,9389); stessa direzione in tutti e tre i fold. Allenando sui soli mondi spostati il guadagno è minore (0,9361) e nel mondo del train si perde 0,0017: scartato. A 3200 alberi il guadagno è simile (0,9348 → 0,9374), ma si resta a 800 per il precedente di v7. Il controllo è in parte circolare, perché i mondi valutati vengono dallo stesso modello di spostamento: `cv_mondo.py --sd 1.2 2.4` misura anche scarti di ampiezza doppia.
- Densità diretta (`scripts/cv_densita.py`, 3 fold, 800 alberi, 19/10): un modello di k = q/v in L1 mediato con q̂ alza S_LWR in 30 corridoi·fold su 30, ma solo da 0,597 a 0,605, e abbassa S_state di 0,0017. Sul totale vale +0,0003: scartato. L'errore su N si concentra nelle celle in coda, dove N = qL/v è grande e piccoli errori di velocità pesano molto.
- S_FD non offre margine: in flusso libero lo scarto dal triangolo è q·|v_f/v − 1|, con v_f pari al quantile 0,99 della velocità osservata. Si azzera solo portando la velocità a v_f, a spese di S_state.

## Risultati

Task 1 locale: S_state con gli stessi conti di `score_task1.py` (verificato: la baseline ufficiale dà 0,6255 in entrambi).

| Data | Esperimento | Task 1 locale (train) | Pubblico |
|---|---|---|---|
| 18/10/2026 | baseline ufficiale del Task 1 (media storica) | 0,6255 su D12_I5_N | — |
| 18/10/2026 | `v1`: T1 interpolazione nel tempo, poi spazio, poi profilo; T2 persistenza dallo stato a T; T4 NNLS con prior (λ = 0,05) sui conteggi dello split | ~0,904 su D12_I5_N | 0,72215 (421°/513) |
| 19/10/2026 | `v3`: come v1, T2 LightGBM per condizione (onset solo T+30), soglie 0,40/0,45 dalla validazione a blocchi | come v1 | **0,83843** (258°/513) |
| 19/10/2026 | `v4`: come v3, T4 minima entropia relativa (KL) dal prior dello split | come v1 | 0,80818: ΔS_ODME ≈ −0,15, scartata |
| 19/10/2026 | `v5`: come v3, T4 NNLS con la media dei tre prior | come v1 | 0,78720: ΔS_ODME ≈ −0,26, scartata |
| 19/10/2026 | `v6`: come v3, T1 LightGBM sui residui dell'interpolazione (800 alberi, 48 caratteristiche) | 0,939 fold 0 (interpolazione 0,905); S_LWR stimato 0,52 → 0,60 | **0,85232** (209°/513) |
| 19/10/2026 | `v7`: come v6, T1 a 3200 alberi | 0,9411 a tre fold contro 0,9393, meglio in 30 corridoi·fold su 30; S_LWR 0,597 → 0,603 | 0,85128 (210°/515): −0,001, non trasferito |
| 19/10/2026 | `v8`: come v6, T2 ongoing allenato su finestre attuali + scelte a caso tra i candidati | ongoing sulle finestre tipo ufficiale +0,05 (3 fold), sulle 40 ufficiali fuori campione 0,857 contro 0,835 | 0,85098 (210°/515): −0,0013, dentro il rumore del Task 2; per regola il Task 2 non si sceglie sulla classifica. Nuovo riferimento |
| 19/10/2026 | `v9`: come v8, T1 allenato sul train più una copia in mondi spostati (randomizzazione del mondo) | mondi spostati 0,9346 → 0,9367, mondo del train 0,9393 → 0,9389; S_LWR nei mondi spostati 0,590 → 0,595 | 0,85050 (212°/519): −0,0005, non trasferito |
| 20/10/2026 | `v10`: come v8, T1 con un secondo modello, allenato sui buchi del Task 2 riprodotti nel train, sui soli bersagli da T a T+24 delle finestre del Task 2; il resto identico a v8 | nei buchi RMSE del flusso 237 → 79, residuo di conservazione 1,69 → 0,81 (3 fold); S_LWR del mese circa +0,04 | **0,85449** (211°/530): +0,0035 su v8, in linea con l'atteso (0,10 × 0,04 ≈ +0,004). Nuovo riferimento |
| 20/10/2026 | `v11`: come v10, T1 con E1 (scarto del livello notturno del mese dal train tolto dagli ingressi e restituito alle velocità previste), stessi modelli | rimascheratura dei mesi veri: S_state +0,0027/+0,0029, S_LWR +0,0018/+0,0020 (validation/private, 20/20); S_FD esatto −0,0005 | **0,85566** (202°/540): +0,0012 su v10, come atteso (≈ +0,0012). Nuovo riferimento |
| 20/10/2026 | `v12`: come v11, modello sui buchi allenato anche sulle copie 1–3 dei buchi simulati | nei buchi RMSE v 8,21 → 7,43, q 78,7 → 72,9 (30/30 e 28/30); pseudo-mesi completi +0,00077 (69/3); S_FD esatto +0,00001 | **0,85606** (201°/543): +0,0004 su v11, come atteso (0,6 × 0,00077 ≈ +0,0005). Nuovo riferimento |
| 20/10/2026 | `v13`: come v12, modello principale con il livello del mese (`v_lib_mese`, `v_lib_scarto`, `v_it_meno_lib`), allenato su train + mondi spostati | pseudo-mesi +0,0006 su v12 nei due mondi; rimascheratura dei mesi veri S_state +0,0010/+0,0011 (19/20), S_LWR +0,0005/+0,0009; S_FD esatto +0,00014 | **0,85664** (199°/543): +0,0006 su v12 (atteso ≈ +0,0004). Nuovo riferimento |
| 20/10/2026 | `v14`: come v13, T1 da T a T+24 delle finestre del Task 2 con la media del modello sui buchi e della rete convoluzionale E2 sulla zona (rete finale allenata sulla CPU, PyTorch) | nei buchi RMSE v 7,43 → 6,82, q 72,9 → 71,6, residuo −3,7%; pseudo-mesi completi +0,00047 su v10c (63/9); S_FD esatto +0,00003 | **0,85689** (202°/546): +0,0003 su v13, come atteso (0,6 × 0,00047 ≈ +0,0003). Nuovo riferimento |
| 21/10/2026 | `v15`: come v14, T2 ongoing allenato su attuali + casuali dei semi 0–5 | ongoing casuali 0,8294 → 0,8423 (3 fold su 3), ufficiali fuori campione 0,8527 → 0,8586; onset identico | 0,85607 (210°/555): −0,0008 su v14 (atteso ≈ +0,002), dentro il rumore del Task 2; per regola il Task 2 non si sceglie sulla classifica. Nuovo riferimento |
| 28/10/2026 | `v16`: come v15, T2 onset dallo stack di LightGBM, XGBoost e CatBoost con il contesto della finestra (`allena_task2_stack.py`, `task2.modello_stack`) | onset attuali 0,8649 → 0,8787 (3 fold su 3, [+0,0078; +0,0189]); ufficiali 0,9169 → 0,9548; mondi spostati +0,024 su LightGBM | 0,85118 (273°/649): −0,0049 su v15 (atteso ≈ +0,002); 26 celle in 10 finestre pubbliche. Fuori dal rumore della CV (≈ 0,5%), causa non trovata (si veda «Task 2: altri motori e stacking») |
| 28/10/2026 | `v17`: come v15, T2 onset con E1 (`task2.modello_e1_onset`): scarto del livello notturno del mese, notti fino a T, tolto dalle velocità di storia e di T; stessi modelli | onset attuali: mondo del train −0,0007 [−0,0025; +0,0012]; mondi spostati +0,0120 [+0,0077; +0,0166] e +0,0067 [+0,0021; +0,0109] | 0,85624 (273°/649): +0,0002 su v15 (atteso ≈ +0,001); 11 celle in 6 finestre, 4 pubbliche. Nuovo riferimento |
| 28/10/2026 | `v18`: come v17, T2 ongoing dallo stack di LightGBM, XGBoost e CatBoost con il contesto della finestra (`task2.modello_e1_onset_stack_ongoing`) | ongoing casuali 0,8423 → 0,8650 (3 fold su 3, [+0,0201; +0,0255]); etichette lisciate +0,019–0,021; mondo spostato invariato; ufficiali 0,8586 → 0,8637 | **0,85998** (261°/653): +0,0037 su v17 (atteso ≈ +0,0027); 452 celle in 63 finestre ongoing, 30 pubbliche. Primo cambio delle ongoing che si trasferisce. Nuovo riferimento e miglior punteggio |
| 28/10/2026 | `v19`: come v18, modello principale del Task 1 con E5 (stime a basso rango del mese tra le caratteristiche; `allena_task1_finale.py --mondi --mese --e5`) | CV a tre blocchi sopra il ramo del mese: parte del totale +0,00105 (25/30), S_state +0,0019, S_LWR +0,0040; mesi veri rimascherati S_state +0,0021 (20/20, 19/20), S_LWR +0,0022 | **0,86106** (259°/657): +0,0011 su v18 (atteso ≈ +0,0006–0,001). Nuovo riferimento e miglior punteggio |

Da v3 a v6 cambia solo il Task 1: +0,0139 in classifica, di cui circa 0,012 attesi da S_state (0,35 × 0,034), quindi ΔS_physics ≈ +0,013.

Da v6 a v7 cambia solo il numero di alberi del Task 1 e la classifica scende di 0,001, contro +0,0012 attesi. Previsioni identiche ricalcolate in locale e sul server, quindi la catena è a posto. Il rumore dei giorni non basta a spiegarlo, perché il guadagno locale è presente in ogni stima da 9 giorni. Resta da capire se sul mese pubblico non sia salito S_state (mesi rigenerati con semi nuovi) o se sia sceso S_physics (≈ −0,011 se S_state è salito come atteso): la classifica non lo dice. Decisione: si resta a 800 alberi (v6) e non si aumenta oltre la capacità del Task 1; le stime locali di S_LWR vanno prese come indicazione di massima.

### Task 1: perché v7 e v9 non si trasferiscono (19/10/2026)

Controlli fatti dopo v9, tutti senza usare la classifica:

- Nei mesi veri S_state sale davvero (`scripts/rimaschera_valutati.py`: si oscura il 10% delle celle osservate di validation e private, due semi, e si ricostruiscono con la catena dell'invio). Rispetto a v6, v9 fa +0,0010 in validation (meglio in 19 corridoi·semi su 20) e +0,0012 in private (20/20); v7 fa 0,0000. L'interpolazione resta 0,031 sotto v6.
- S_FD si calcola esatto senza verità (`metrica.s_fd_valutato`, `scripts/fd_valutati.py`): v3 0,9864, v6 0,9880, v7 0,9878, v9 0,9878 in validation. Non spiega nulla.
- La nostra stima della parte fisica è calibrata: sulla baseline ingenua (media storica per tratto e ora della settimana) dà S_physics 0,3243 su D12_I5_N contro 0,3242 ufficiale, e S_LWR circa 0 come l'ufficiale. Un termine noto senza rumore di velocità darebbe alla risposta perfetta 0,75, non 0,96: ipotesi scartata.
- In flusso libero la velocità vera di ogni tratto è una costante del mese più rumore bianco: deviazione standard del livello e del rumore coincidono (0,91 e 0,91 km/h su D12_I5_N, 1,74 e 1,74 su D7_I405_N), nessuna dipendenza dall'ora, correlazione nulla con il flusso.
- **La causa: i buchi del Task 2.** Nei mesi valutati, per ognuna delle 10 finestre del Task 2, gli organizzatori oscurano su tutto il corridoio le celle eleggibili non bersaglio da T+5 a T+90 minuti (orizzonte più un'ora; MASK_SPEC.md), e la storia della finestra (T−60 … T−5) è tutta osservata e senza bersagli. I buchi di validation si ricostruiscono tutti così dalle finestre. Sono l'1,4% delle celle eleggibili, toccano il 16–26% dei tratti·giorni e partono all'inizio delle code; nel train non ce ne sono. Il valutatore del Task 3 vi mette la verità. Lì dentro cade l'1,6% dei bersagli, e lì le versioni divergono: flusso v7 − v6 di 200 vph in media contro 14 fuori, v3 − v6 di 837 contro 79; quell'1,6% pesa il 12–13% di tutta la differenza su N tra le versioni.
- Buchi riprodotti nel train (`scripts/cv_buchi.py`, 25 finestre per corridoio e blocco dalla regola pubblicata, tre fold; prima versione, con la storia della finestra mascherata come il resto): dentro i buchi LightGBM com'è oggi sbaglia il flusso molto più dell'interpolazione (RMSE per corsia 237 a 800 alberi e 283 a 3200, contro 96), e il residuo di conservazione nella regione delle finestre vale 1,46 e 1,64 volte Σ|ΔN| contro 0,87. Su un mese con 10 finestre costa circa 0,09 di S_LWR a 800 alberi. Allenando anche sulle righe con i buchi l'RMSE del flusso scende a 79 e il residuo a 0,75, meglio dell'interpolazione in 29 corridoi·fold su 30; il guadagno di S_LWR sul mese è +0,042 (30/30, minimo +0,024) e l'RMSE del flusso nei buchi migliora in 30/30. Con queste perdite il conto torna con la classifica: v3 → v6 atteso +0,0135 (osservato +0,0139), v6 → v7 circa −0,002 (osservato −0,001).
- Seconda versione della simulazione, fedele ai mesi valutati: storia della finestra (T−60 … T−5) tutta osservata e senza bersagli, bersagli da T a T+24. Tre fold: RMSE del flusso nei buchi 237 (LightGBM com'è oggi, 800 alberi) contro 79 (allenato anche sui buchi) e 95 (interpolazione); residuo di conservazione nella regione 1,69 contro 0,81 e 0,94. Il modello finale sui buchi (`allena_task1_finale.py --buchi --nome finale_buchi`, 800 alberi) usa le righe di questa versione.
- Lo stesso modello usato ovunque costa un poco sui giorni senza buchi (S_state −0,0006, meglio in 3/30; S_LWR −0,003, meglio in 5/30). Quindi v10 usa due modelli (`task1.ricostruisci_lgb_buchi`): quello di v6, invariato, su tutti i bersagli tranne quelli da T a T+24 delle finestre del Task 2 dello split (`window_index.csv`), dove usa il modello allenato anche sui buchi. La regione copre il 100% dei buchi veri sugli 8 corridoi con il Task 2 e il 2–4% dei bersagli; D12_I405_N e D12_I405_S non hanno finestre e restano come v6. Fuori dalla regione le previsioni sono identiche a v8 (verificato).
- Esito di v10 (20/10/2026): 0,85449 contro 0,85098 di v8, +0,0035 con il solo cambio del Task 1 sul 2,6% delle righe. La stima era stata fatta prima dell'invio sul solo train (+0,04 di S_LWR, cioè circa +0,004 sul totale): è il primo cambio del Task 1 dopo v6 la cui stima locale si trasferisce. Resta non verificato perché v9 non si trasferisca anche fuori dai buchi.

### Task 1: esperimenti della revisione v2 (20/10/2026)

Proposte di `REVISIONE_AVVERSARIALE_v2.md`; E6 ed E7 (Task 2) sono più sotto.

**Più esempi per il modello sui buchi** (`cv_buchi.py copie --copie 3`, poi `cv --copie 3`). Copie 1–3 dei buchi simulati, ognuna con un'altra finestra a caso in ogni giorno che ne ha una e mai la stessa (6.160 finestre in più contro 750), solo per allenare; si valuta sempre sulla copia 0. Rispetto al modello sui buchi di v10:

| Allenamento | RMSE v nei buchi | RMSE q nei buchi | E_regione | ΔS_LWR del mese |
|---|---:|---:|---:|---|
| copia 0 (v10) | 8,21 | 78,7 | 0,815 | — |
| copie 0–1 | 7,64 | 74,1 | 0,755 | +0,0027 (29/30, minimo −0,0010) |
| copie 0–3 | **7,43** | **72,9** | **0,737** | **+0,0035** (30/30, minimo +0,0002) |

La curva sale ancora. La misura mette la verità fuori dalla regione delle finestre (revisione v2 §2.1): va confermata sui pseudo-mesi.

**Pseudo-mesi completi** (`scripts/cv_pseudomesi.py`). Ogni mese del train diventa un mese valutato:
- 10 finestre per corridoio come in `window_index.csv`: 5 onset, le prime ammesse; 5 ongoing scorrendo i giorni e, nel giorno, a caso. Nei mesi valutati cadono nei primi giorni (mediana 3–4, massimo 20), e così queste;
- storia protetta e buchi; i bersagli del train nella storia si tolgono, perché nei mesi valutati lì non ce ne sono;
- tutti i bersagli del mese ricostruiti con la procedura dell'invio (un ramo fuori dalla regione, uno dentro);
- S_state, S_FD e S_LWR sul mese intero (`s_fisica_stima`) e la parte del totale 0,35·S_state + 0,05·S_FD + 0,10·S_LWR; confronti appaiati con intervallo bootstrap sui 90 corridoi·mesi. Anche in mondi spostati (`--mondo 1.2`, un mondo per corridoio e mese).

Parte del totale (0,35·S_state + 0,05·S_FD + 0,10·S_LWR) rispetto a v10, intervallo al 90%, corridoi·mesi meglio/peggio (fuori/dentro = ramo fuori e dentro la regione):

| Metodo | Mondo del train | Mondi spostati (1,2 km/h) |
|---|---|---|
| v8 (std/std) | −0,0060 [−0,0066; −0,0054] 0/72 | −0,0060 0/72 |
| v10c (std/buchi3) | +0,00077 [+0,00064; +0,00088] 69/3 | +0,00074 70/2 |
| v10c + E1 | +0,00076 79/11 | +0,00273 [+0,00261; +0,00285] 90/0 |
| v9c (mondi/buchi3) | +0,00069 65/25 | +0,00179 89/1 |
| mesec (mese/buchi3) | +0,00131 [+0,00118; +0,00146] 87/3 | +0,00321 90/0 |
| mesec + E1 | +0,00132 87/3 | +0,00336 [+0,00322; +0,00351] 90/0 |

- Taratura sulla classifica: da v8 a v10 i pseudo-mesi danno +0,0060 contro +0,0035 osservati; E1 nei mondi spostati +0,0020 contro +0,0012 osservati (v11). Stesso fattore, circa 0,6: i pseudo-mesi ordinano bene ma gonfiano l'effetto di circa 1,7 volte.
- Le copie dei buchi reggono sul mese intero (+0,00077, 69/3): in classifica circa +0,0005.
- Il livello del mese come caratteristiche, con i mondi spostati, aggiunge circa +0,0006 anche sopra E1 e anche nel mondo del train: non viene dallo spostamento ma dal togliere il rumore in flusso libero (il modello tira verso il livello del mese al posto dell'interpolazione tra due vicini rumorosi; la seconda parte di E1 nella revisione).
- Senza E1 la randomizzazione del mondo (v9c) recupera nei mondi spostati meno della metà di quanto recupera E1.

Livello del mese sui mesi veri (`rimaschera_valutati.py --versioni v10 v13 --e1`; v13 = modello principale `finale_mese`, allenato con `--mondi --mese`): v13 con E1 rispetto al ramo principale di v12 (v10 con E1), appaiato per corridoio e seme:

| | validation | private |
|---|---|---|
| ΔS_state | +0,00105 (19/20, minimo −0,0007) | +0,00100 (19/20, minimo −0,0002) |
| ΔS_LWR | +0,0005 (17/20) | +0,0009 (16/20) |

Atteso sul totale circa +0,0004, come dai pseudo-mesi (0,6 × 0,0006). Senza E1 v13 fa quasi lo stesso (+0,0037 di S_state su v10): il modello con il livello del mese e i mondi spostati ha imparato da sé a togliere lo scarto. E1 resta per il modello sui buchi, che non ha queste caratteristiche.

**E1: togliere lo scarto di livello del mese e restituirlo** (`task1.equivariante`, revisione v2 §5). Prima la verifica sui mesi veri: per fasce di occupazione, lo spostamento di velocità di ogni tratto in validation e private rispetto al train, regredito sullo scarto del suo livello notturno (1:00–4:00, sempre flusso libero: nessuna cella sotto 0,9·v_f):

| Occupazione (quantili del train) | v/v_f | Pendenza | Correlazione |
|---|---:|---:|---:|
| 0–0,50 | 0,999 | 0,996 | 0,999 |
| 0,50–0,80 | 1,000 | 0,996 | 0,998 |
| 0,80–0,90 | 1,000 | 0,998 | 0,994 |
| 0,90–0,95 | 0,999 | 0,985 | 0,923 |
| 0,95–0,98 | 0,705 | 0,780 | 0,214 |
| 0,98–1 | 0,344 | 0,162 | 0,042 |

Lo scarto è tutto in flusso libero e sparisce in coda, come in `cv_mondo.sposta`. E1 quindi stima lo scarto δ di ogni tratto (livello notturno del mese meno quello del train), lo toglie dalle velocità osservate con il peso del flusso libero (pieno sopra 0,95·v_f, nullo sotto 0,8·v_f), ricostruisce con i modelli invariati e restituisce δ alle velocità previste. Niente da allenare. Nel train δ è rumore (0,06 km/h).

Rimascheratura dei mesi veri (`rimaschera_valutati.py --versioni v10 v9 --e1`, 10% delle celle osservate, due semi), rispetto a v10:

| Metodo | ΔS_state validation | ΔS_state private | ΔS_LWR validation | ΔS_LWR private |
|---|---|---|---|---|
| v9 | +0,0010 (19/20) | +0,0012 (20/20) | +0,0008 (15/20) | +0,0011 (19/20) |
| **v10 + E1** | **+0,0027 (20/20)** | **+0,0029 (20/20)** | **+0,0018 (19/20)** | **+0,0020 (20/20)** |
| v9 + E1 | +0,0022 (20/20) | +0,0024 (20/20) | +0,0017 (18/20) | +0,0020 (20/20) |

L'interpolazione con e senza E1 dà gli stessi numeri, come deve. Atteso sul totale circa +0,0012 (la stima di S_LWR sulle celle osservate è schiacciata verso 1). Diventa `v11`: v10 con E1, stessi modelli. In classifica 0,85566 contro 0,85449 (+0,0012): la stima si trasferisce. S_FD esatto scende di 0,0005 (0,98821 → 0,98767 in validation), −0,00003 sul totale: le velocità previste in flusso libero si allontanano da v_f come la verità del mese.

**E2/E3: ricostruire la zona di una finestra come una sequenza** (`scripts/e2_sequenze.py`, GPU). Una rete convoluzionale piccola (8 blocchi residui con dilatazioni, 64 canali) vede tutto il corridoio da T−12 a T+36 e prevede i residui di v e q dei bersagli da T a T+24, rispetto alla stessa base dei modelli LightGBM. Ingressi: valori osservati, maschere di osservazione e dei bersagli, base in un canale a parte, distanze dalle osservazioni, ora, statici. E3 aggiunge alla perdita λ·Σ|Δe_N| / Σ|ΔN| (e_N = errore su N nei bersagli, 0 altrove). Stesse finestre e fold di `cv_buchi.py`, contro LightGBM sui buchi con le copie 0–3 sulle stesse celle:

| Metodo | RMSE v nei buchi | RMSE q nei buchi | Residuo | Meglio di LightGBM (v, q, residuo; su 30) |
|---|---:|---:|---:|---|
| LightGBM sui buchi, copie 0–3 | 7,43 | 72,9 | — | — |
| rete, λ = 0 (E2) | 6,98 | 76,2 | +7,7% | 19, 5, 5 |
| rete, λ = 1 (E3) | 7,75 | 86,1 | −12,4% | 8, 0, 26 |
| media rete E2 + LightGBM | **6,82** | **71,6** | **−3,7%** | 25, 20, 23 |
| media rete E3 + LightGBM | 7,24 | 74,9 | −10,3% | 16, 4, 29 |

- Da sola la rete non basta: E2 migliora la velocità e peggiora il flusso; E3 riduce il residuo su N ma perde in RMSE (sul totale circa +0,0001: il caso «fermarsi» della revisione).
- La media con LightGBM migliora tutte e tre le misure: i due modelli sbagliano in modo diverso. Valore stimato piccolo (circa +0,0005 sul totale: i buchi sono il 2,6% dei bersagli).
- Sui pseudo-mesi completi, rispetto a v10c (std/buchi3): rete E2 da sola dentro le regioni +0,00015 (37/35); media rete E2 + LightGBM **+0,00047** [+0,00038; +0,00057] (63/9; S_state +0,00096, S_LWR +0,00139); media rete E3 + LightGBM +0,00034 (63/9; S_LWR +0,0028). In classifica circa +0,0003: la media E2 resta candidata, al prezzo di PyTorch nel pacchetto finale.
- PyTorch 2.14.1 è nel gruppo facoltativo `gpu` di `pyproject.toml`; l'ambiente dell'invio non cambia finché la rete non entra in un invio.
- In v14 (`task1.ricostruisci_lgb_buchi_rete_e1`) la rete finale è allenata su tutte le 6934 zone del train e delle copie (`e2_sequenze.py finale --lam 0`, 30 epoche sulla CPU, perdita finale 0,0253 contro 0,027–0,029 dei fold) e congelata in `outputs/invii/v14/modelli/task1/rete_buchi.pt`, registrata nel manifesto. Rispetto a v13 cambiano solo i 178.320 bersagli da T a T+24 delle finestre (|Δv| medio 0,70 km/h, |Δq| medio 59 veicoli/h, nessun valore fuori dagli intervalli di v13); S_FD esatto +0,00003/+0,00002.

Non fatti: E4 (interpolazione lungo la propagazione) e il selettore tra esperti (§12). La revisione li tiene solo se servono; la media di E2 con LightGBM copre già il caso di due esperti con errori diversi.

**E5: completamento a basso rango del mese** (28/10/2026, `scripts/e5_basso_rango.py`; un notebook pubblico riporta +0,003 in classifica). Per corridoio e mese, la matrice (giorno × istante) × tratto delle osservazioni eleggibili dello strato mascherato si completa a rango r con minimi quadrati alternati (cresta 1), per velocità e flusso per corsia. Il solutore è corretto: su dati sintetici di rango 3 con rumore 30 l'errore sulle celle tenute da parte è 30,4 al rango giusto. Sui dati veri, tenendo da parte il 10% delle celle osservate di due mesi per corridoio:

| Rango | 1 | 2 | 4 | 8 | 12 |
|---|---:|---:|---:|---:|---:|
| RMSE velocità (km/h) | 9,95 | 9,32 | **8,67** | 8,94 | 9,26 |
| RMSE flusso per corsia | 60,9 | **58,0** | 69,4 | 116,8 | 167,5 |

Il modello del Task 1 sta a circa 1,7 km/h e 33 veicoli/h: i dati non sono a basso rango in questa forma. Le code rompono la velocità e le fluttuazioni del flusso sono reali e correlate tra tratti vicini (misurato il 18/10), quindi l'interpolazione locale le segue meglio di qualunque profilo del mese. Come caratteristica (rango 4 per v, 2 per q) va alla CV a tre blocchi; regola dichiarata prima: entra solo se la parte del totale sale in almeno 25 corridoi·fold su 30 e di almeno +0,0005.

| CV a tre blocchi, 800 alberi | S_state | S_LWR | RMSE v | RMSE q | Parte del totale | Δ, corridoi·fold meglio |
|---|---:|---:|---:|---:|---:|---|
| std | 0,93929 | 0,59653 | 1,661 | 32,39 | 0,38840 | — |
| std + E5 | 0,94141 | 0,60151 | 1,595 | 31,49 | 0,38964 | +0,00124, 25/30 (minimo −0,0026) |
| std + mese (ramo dell'invio) | 0,94008 | 0,59867 | 1,624 | 32,39 | 0,38890 | — |
| std + mese + E5 | 0,94194 | 0,60267 | 1,570 | 31,48 | 0,38995 | +0,00105, 25/30 (minimo −0,0027) |

Da solo il basso rango stima male, ma accanto all'interpolazione dà al modello un riferimento diverso: il profilo del tratto nel mese, che il livello del mese copre solo in piccola parte (il guadagno sul flusso è lo stesso, −0,9 veicoli/h). Peggiora solo D12_I5_N (−0,0005). Rimascheratura dei mesi veri (`rimaschera_valutati.py --versioni v18 v19 --e1`): v19 con E1 contro v18 con E1, S_state +0,0021 in validation (20/20) e private (19/20), S_LWR +0,0022 (17/20, 15/20). Entra solo nel ramo principale; le righe con i buchi non hanno E5, come per il livello del mese. In produzione `caratteristiche.calcola(e5=True)` completa il mese dello split (strato mascherato, dopo E1). Diventa v19: +0,0011 in classifica.

### Task 1: XGBoost accanto a LightGBM (20/10/2026)

Domanda: la media pesata di LightGBM e XGBoost batte LightGBM da solo? `cv_pseudomesi.py modelli --motore xgb --rami mese buchi3`: XGBoost (`xgboost-cpu` 3.4.1, gruppo facoltativo `xgb`) sugli stessi rami, righe, fold e caratteristiche dei modelli di v14, con una configurazione standard non regolata (alberi a profondità 8 invece che a 127 foglie, `min_child_weight` 200, 800 alberi, η 0,05, sottocampioni 0,8). RMSE dei residui fuori dal fold, per peso di LightGBM nella media (v in km/h, q per corsia in veicoli/h):

| Ramo, righe | 100% (v14) | 75% | 50% | 25% | 0% (solo XGBoost) | 50% − 100% per fold |
|---|---:|---:|---:|---:|---:|---|
| `mese`, train: q | 33,464 | **33,422** | 33,441 | 33,521 | 33,662 | −0,030 −0,015 −0,024 |
| `mese`, train: v | 1,739 | **1,735** | 1,738 | 1,747 | 1,763 | −0,003 −0,002 +0,001 |
| `buchi3`, buchi (copia 0): q | **65,303** | 65,304 | 65,449 | 65,735 | 66,162 | +0,160 +0,102 +0,169 |
| `buchi3`, buchi (copia 0): v | **6,219** | 6,231 | 6,261 | 6,307 | 6,370 | +0,033 +0,048 +0,046 |

Copie 1–3 dei buchi come la copia 0; le righe dei mondi spostati come quelle del train.

- XGBoost da solo è peggiore ovunque.
- Nel modello principale la media migliora pochissimo: −0,13% su q e −0,2% su v al 75%, con un fold su tre contro per v. Sul totale vale meno di +0,0001, sotto il rumore e sotto la soglia che conta.
- Nei buchi, dove la rete aveva aiutato, ogni peso di XGBoost peggiora, in tutti i fold.
- Conclusione: non entra nell'invio e non serve la valutazione sui pseudo-mesi. Due implementazioni di alberi con gli stessi ingressi sbagliano quasi allo stesso modo. Errori diversi vengono da informazioni diverse, come la zona intera vista dalla rete E2, non da un secondo motore di alberi.
- Restano nel repository il codice (`--motore xgb`, metodi `ramo@x` e `ramo@NN` in `valuta`) e, sul server, i modelli e le previsioni fuori dal fold (`outputs/task1/pseudomesi/oof_{mese,buchi3}.parquet`).

### Task 2: validazione a blocchi (19/10/2026)

`scripts/cv_task2.py`: 4.196 finestre ricostruite sui 9 mesi del train con la regola di selezione, tre blocchi temporali (giorni 0–91, 92–182, 183–272), un modello LightGBM per condizione allenato sugli altri due blocchi. La frequenza storica di coda esclude il blocco della finestra. IoU medio per corridoio, poi sui corridoi:

| Condizione | Persistenza da T | LightGBM, soglia scelta | Fold peggiore | Plateau della soglia |
|---|---:|---:|---:|---|
| `queue_onset` (solo T+30) | 0,002 | 0,865 (0,40) | 0,861 | 0,30–0,45 entro 0,005 |
| `queue_ongoing` | 0,815 | 0,914 (0,45) | 0,912 | 0,40–0,50 entro 0,005 |

Soglia scelta al massimo del fold peggiore; le curve sono piatte attorno, quindi la scelta non dipende da un fold. Se nessuna cella supera la soglia si prevede in coda il tratto più probabile (la coda nell'orizzonte c'è per costruzione). Il modello finale (`scripts/allena_task2_finale.py`) usa le stesse righe e gli stessi parametri su tutti i blocchi. In produzione la storia viene da `window_history.parquet` (identica allo strato vero del train sulle 30 finestre ufficiali controllate), la riga a T dallo strato mascherato dello split, la frequenza da tutto il train: nessun dato dopo T.

### Task 2: controlli della revisione (19/10/2026)

`scripts/cv_task2_varianti.py`, stesse finestre e parametri, IoU alle soglie dell'invio:

| | blocchi (come sopra) | frequenze annidate | senza frequenza | progressiva 0 → 1 | progressiva 0–1 → 2 |
|---|---:|---:|---:|---:|---:|
| onset | 0,865 | 0,869 | 0,831 | 0,825 | 0,860 |
| ongoing | 0,914 | 0,911 | 0,902 | 0,873 | 0,916 |

- Le righe di allenamento usavano frequenze calcolate anche sul blocco valutato. Togliendolo (annidata) la stima si sposta di ±0,004, in direzioni opposte: nessun ottimismo misurabile. La frequenza serve (senza, l'onset perde 0,034).
- Allenando solo sul passato con 6 mesi (0–1 → 2) si ritrova la stima a blocchi dello stesso blocco; con 3 mesi (0 → 1) si perde, come atteso con metà dei dati. L'invio ne usa 9.
- Soglie scelte sul passo 0 → 1 e misurate una volta sul blocco 2: 0,862 onset e 0,917 ongoing; con le soglie dell'invio 0,860 e 0,916.
- Finestre ufficiali del train ritrovate tra le nostre (46 su 80), previsioni fuori fold: onset 0,915 (39 finestre) contro 0,875 sulle nostre; ongoing 0,866 (7 finestre) contro 0,926. Insieme alla persistenza (0,760 sulle 40 ongoing ufficiali, 0,815 sulle nostre) dice che le ongoing ufficiali sono più difficili delle ricostruite: in classifica ci si aspetta circa 0,85 sulle ongoing, non 0,91. Spiega in parte perché da v1 a v3 la classifica è salita meno della stima (+0,116 contro +0,144).

### Task 2: come sono scelte le ongoing ufficiali (19/10/2026)

- Le 80 finestre ufficiali del train stanno nei primi giorni (0–19), 5 per condizione e corridoio; la copertura della storia coincide con la nostra al decimale.
- Onset: 39 su 40 sono la prima origine ammessa in ordine di tempo, come la nostra selezione.
- Ongoing: solo 7 su 40 coincidono. Tutte soddisfano la nostra regola, ma cadono in posizioni sparse tra i candidati dello stesso giorno (quantile mediano 0,58; nessun criterio semplice, tra dimensione della coda, IoU della persistenza, copertura e ora, le mette al primo posto). Si comporta come una scelta casuale; la nostra prendeva sempre la prima origine, cioè l'inizio degli episodi.
- `finestre.ongoing_casuali` sceglie a caso tra i candidati, a 360 minuti l'una dall'altra: la persistenza sulle casuali vale 0,748, quasi come sulle ufficiali (0,760), contro 0,835 sulle attuali. La distanza unica con le onset lasciava a D7_I10_W 4 ongoing in tutto il train.
- `scripts/cv_task2_selezione.py` (le ufficiali sono tutte nel blocco 0, quindi un modello sui blocchi 1–2 le prevede fuori campione):

| Allenamento → | attuali | casuali | unione |
|---|---:|---:|---:|
| 40 ongoing ufficiali | 0,835 | 0,836 | 0,857 |
| casuali, fold 0 / 1 / 2 | 0,789 / 0,774 / 0,762 | 0,836 / 0,827 / 0,817 | 0,836 (fold 0) |
| attuali, fold 0 | 0,913 | 0,881 | 0,914 |

Il modello ongoing finale si allena sull'unione; soglie invariate (curve piatte tra 0,40 e 0,50).

### Task 2: decisione orientata all'IoU (E6, E7; 20/10/2026)

`scripts/cv_task2_decoder.py` (proposta della revisione v2) rifà le probabilità fuori campione con la ricetta di v8: un modello per condizione, 400 alberi, ongoing allenato su finestre attuali più casuali. Riproduce i riferimenti: onset 0,8649 contro 0,865, ongoing ufficiali 0,8527 contro 0,857. Sulle stesse probabilità confronta la soglia fissa (0,40 onset a T+30, 0,45 ongoing) con:
- top-k per finestra, con k che massimizza l'IoU attesa, approssimata come rapporto di attese oppure stimata per Monte Carlo da Bernoulli indipendenti;
- le stesse su probabilità calibrate con l'isotonica sulle previsioni interne;
- per le ongoing, una decodifica di Viterbi per tratto con penalità λ sui cambi di stato (E7).

Calibrazione e λ vengono solo da modelli interni. La regola di scelta era dichiarata prima: fold peggiore più alto sulle ricostruite, con la media sopra la soglia. Intervalli dal bootstrap per giorno.

| Condizione, finestre | Soglia | Migliore alternativa | Δ medio [intervallo] | Fold peggiore |
|---|---:|---|---|---|
| onset, attuali (2.102) | 0,8649 | top-k approssimato 0,8672 | +0,0023 [−0,0000; +0,0045] | 0,8611 → 0,8581 |
| onset, ufficiali (40) | 0,9169 | top-k Monte Carlo calibrato 0,9174 | +0,0008 [−0,064; +0,045] | — |
| ongoing, casuali con tre semi (9.752) | 0,8294 | top-k Monte Carlo 0,8274 | −0,0020 [−0,0040; −0,0000] | 0,8240 → 0,8225 |
| ongoing, attuali (2.094) | 0,8852 | top-k Monte Carlo 0,8907 | +0,0077 [−0,032; +0,045] | 0,8573 → 0,8631 |
| ongoing, ufficiali (40) | 0,8527 | top-k Monte Carlo calibrato 0,8582 | +0,0059 [−0,0016; +0,0167] | — |

- Onset: top-k guadagna poco in media ma abbassa il fold peggiore; sulle ufficiali va da −0,025 a +0,001.
- Ongoing: sulle casuali, che imitano la selezione ufficiale, ogni top-k perde da 0,002 a 0,004, e per tre varianti su quattro l'intervallo esclude lo zero. Il guadagno sulle attuali (inizi degli episodi) e sulle 40 ufficiali non si ripete lì.
- Viterbi: λ interno = 0 in tutti i fold, perché l'IoU interna scende con λ (0,8476 a λ = 0, 0,8407 a λ = 2 nel fold 0).
- Limite: le soglie 0,40 e 0,45 vengono dagli stessi fold, quindi il confronto favorisce un po' la soglia; non spiega però le perdite sulle casuali.

Decisione: la soglia fissa resta, nessuna regola nuova per il Task 2.

### Task 2: le code dello stesso mese prima di T (20/10/2026)

Ipotesi: nei mesi valutati la quota di coda per tratto cambia di 2,5 punti, contro 1,3 tra i mesi del train, e la frequenza del modello viene tutta dal train. La regola ammette ogni dato ≤ T, quindi anche i giorni precedenti dello stesso mese.

`scripts/cv_task2_mese.py` aggiunge sei colonne per finestra e tratto:
- quota in coda e numero di celle osservate nei giorni precedenti del mese, da T a T+12;
- la stessa quota meno la frequenza del train;
- quota e numero su tutto il mese prima di T;
- coda il giorno prima.

Le osservazioni vengono dallo strato mascherato del train, senza le celle oscurate attorno alle 10 finestre valutate del mese. Per il resto ricetta, righe e fold sono quelli di E6, e la regola di decisione era dichiarata prima dei risultati.

| Finestre | v8 | Con il mese | Δ [bootstrap per giorno] |
|---|---:|---:|---|
| onset, attuali (2.102) | 0,8649 | 0,8572 | −0,0078 [−0,0128; −0,0030], peggio in 3 fold su 3 |
| onset, primi 7 giorni del mese (487) | 0,8695 | 0,8520 | −0,0177 [−0,0288; −0,0059] |
| ongoing, casuali con tre semi (9.752) | 0,8294 | 0,8276 | −0,0018 [−0,0037; +0,0001] |
| ufficiali onset / ongoing (40 + 40) | 0,9169 / 0,8527 | 0,9050 / 0,8516 | −0,012 / −0,001 |

- Il modello onset usa molto le colonne nuove (circa il 16% del guadagno), ma fuori campione peggiora.
- Le code che cambiano da un mese all'altro vengono dagli incidenti, che non si ripetono. I primi giorni del mese non dicono dove si formeranno le prossime; la frequenza del train, stimata su molti mesi, lo dice meglio.
- Decisione: scartata, nessuna variante cercata dopo.

### Task 2: dove si sbaglia e più finestre di allenamento (20/10/2026)

Errori sulle ongoing casuali alla soglia dell'invio (9.752 finestre, probabilità fuori campione di E6):

| Errore | Celle |
|---|---:|
| prevista in coda su un tratto in coda a T, la coda si è sciolta | 43.400 |
| non prevista su un tratto nuovo, la coda si è allungata | 26.200 |
| prevista sciolta, la coda è rimasta | 11.600 |
| prevista su un tratto nuovo, lì non c'era | 12.700 |

Si sbagliano lo scioglimento e l'allungamento. Riempire nello spazio i buchi delle code previste non serve: le code vere ne hanno di più (199.000 buchi di 1–2 tratti, contro 56.000 nelle previste).

`scripts/cv_task2_piu_finestre.py`: più finestre di allenamento, come le copie dei buchi nel Task 1. Parametri, fold, righe di valutazione e regola di decisione sono quelli di E6 e di `cv_task2_mese.py`.

| Allenamento | Valutazione | v8 | Nuovo | Δ [bootstrap per giorno] |
|---|---|---:|---:|---|
| onset: attuali + 16.447 origini con la storia senza coda e una coda in un passo qualunque | onset attuali (2.102) | 0,8649 | 0,8388 | −0,026 [−0,035; −0,017] |
| | onset ufficiali (40) | 0,9169 | 0,8751 | −0,043 |
| ongoing: attuali + casuali dei semi 0–2 | ongoing casuali (9.752) | 0,8294 | 0,8395 | +0,0100 [+0,0081; +0,0118], 3 fold su 3 |
| | ongoing ufficiali (40) | 0,8527 | 0,8526 | +0,0002 [−0,005; +0,005] |
| ongoing: attuali + casuali dei semi 0–5 | ongoing casuali (9.752) | 0,8294 | **0,8423** | **+0,0128** [+0,0105; +0,0153], 3 fold su 3 |
| | ongoing primi 7 giorni (2.234) | 0,8356 | 0,8490 | +0,0139 [+0,0090; +0,0189] |
| | ongoing ufficiali (40) | 0,8527 | **0,8586** | +0,0060 [+0,0001; +0,0134] |

- Onset: le origini con la coda prima di T+30 sono un'altra distribuzione e peggiorano; scartate. Quelle con la coda proprio a T+30, come le ufficiali, sono solo 2.332 contro 2.102: non c'è altro da aggiungere.
- Ongoing: l'effetto cresce con i dati e si attenua (da 1 a 3 semi +0,010, da 3 a 6 +0,003), quindi è un effetto dei dati e non del caso. Con 6 semi migliorano anche le ufficiali. Ci si ferma a 6: oltre il guadagno è piccolo e si darebbe più peso alle code lunghe, che hanno molte più origini candidate.
- La soglia 0,45 resta sul plateau (0,8423 contro 0,8427 a 0,50, dentro le differenze tra fold): invariata.
- Sul totale circa +0,002 (metà delle finestre, peso 0,30). Entra in v15 (`allena_task2_finale.py --semi 6`).
- In classifica (21/10): v15 0,85607 contro 0,85689 di v14, −0,0008, cioè circa −0,005 sull'IoU medio delle ongoing pubbliche (40 finestre, 30 delle quali toccate da v15). Sulle finestre attuali, le più simili alle ufficiali, il bootstrap locale va da −0,032 a +0,039: lo scarto sta nel rumore. È il secondo cambio delle ongoing che scende in classifica (v8 −0,0013): due su due, non ancora un segnale (1 caso su 4 per puro caso), ma da tenere presente nella scelta dei due invii finali.

### Task 2: altri motori e stacking (28/10/2026)

Domande dell'utente: XGBoost o CatBoost accanto a LightGBM? Uno stack, con la previsione del primo LightGBM tra le caratteristiche di un secondo? Dove è debole il modello? `scripts/cv_task2_stack.py`, ricetta di v15 al primo livello, stessi tre blocchi; stack annidato (le righe di allenamento del secondo livello ricevono la probabilità di un modello allenato sul solo terzo blocco). Regola di scelta in testa allo script, scritta prima dei risultati. Per ora solo l'onset; le ongoing (26 milioni di righe) sono in calcolo.

| Onset, 2.102 finestre attuali, soglia 0,40 | IoU | Δ su LightGBM [bootstrap 95%] | Fold peggiore | Mondi spostati 1,2 km/h (due mondi) |
|---|---:|---|---:|---|
| LightGBM (v15) | 0,8649 | — | 0,8611 | 0,8522 / 0,8576 |
| XGBoost | 0,8582 | −0,0067 | 0,8504 | 0,8422 / 0,8513 |
| media LightGBM + XGBoost | 0,8647 | −0,0002 [−0,0036; +0,0034] | 0,8631 | 0,8516 / 0,8583 |
| CatBoost | 0,8696 | +0,0047 [−0,0006; +0,0097] | 0,8642 | 0,8665 / 0,8690 |
| media LightGBM + CatBoost | 0,8736 | +0,0087 [+0,0045; +0,0132] | 0,8661 | 0,8669 / 0,8693 |
| stack: caratteristiche + p di LightGBM | 0,8593 | −0,0056 [−0,0113; −0,0002] | 0,8556 | — |
| stack con contesto (p dei vicini, rango, massimo, somma nella finestra) | 0,8721 | +0,0072 [+0,0011; +0,0128] | 0,8677 | 0,8728 / 0,8770 |
| **stack con le p dei tre motori e contesto** | **0,8787** | **+0,0137 [+0,0078; +0,0189]** | **0,8728** | **0,8766 / 0,8812** |

- XGBoost sbaglia come LightGBM, come nel Task 1; CatBoost (alberi simmetrici) sbaglia in modo diverso e la media rende.
- Lo stack semplice peggiora: con gli stessi ingressi il secondo modello non ha nulla di nuovo e riceve una probabilità più rumorosa. Rende con il contesto della finestra, che il primo modello, cella per cella, non vede.
- Lo stack dei tre motori batte la media LightGBM + CatBoost di +0,0051 [+0,0004; +0,0095] (un fold su tre contro) e lo stack con il solo LightGBM di +0,0067 (3/3). Sulle 40 ufficiali 0,9548 contro 0,9169. La soglia sta sul plateau (0,8775–0,8809 tra 0,30 e 0,55).
- Mondi spostati (`mondo_righe`, `mondo`: velocità di ogni tratto + δ ~ N(0; 1,2 km/h) in flusso libero, come `cv_mondo.sposta`, solo sui blocchi valutati): LightGBM perde 0,007–0,013, lo stack quasi nulla, e il suo vantaggio sale a +0,024. Il cambio di mondo così modellato non spiega il calo di v16; dice invece che il LightGBM dell'onset è fragile allo spostamento, come nel Task 1 prima di E1.
- Fasce (`debolezze`): il punto debole è la coda che nasce in un tratto dove storicamente non si forma (frequenza < 0,02: 89 finestre, IoU 0,15–0,20 per tutti i metodi; 233 falsi negativi contro 36 celle giuste). A T quei tratti non rallentano (velocità ≥ 0,9·v_f): probabilmente incidenti, senza informazione da sfruttare. Corridoi più deboli: D12_I5_S (0,76–0,79), D7_I10_W (0,73–0,76), D7_I405_S (0,80–0,84). Nessun motore vince stabilmente in una fascia: un commutatore per fascia non ha appigli.
- Le celle cambiate dallo stack non sono vicine alla soglia (le aggiunte giuste più di 6 km/h sotto, quelle sbagliate più di 6 km/h sopra): il rumore delle etichette osservate non c'entra.
- v16 in classifica: −0,0049. Controlli: la funzione dell'invio, con i modelli del fold 0, riproduce la CV a 3·10⁻⁸; la verifica online passa (48/48); v16 differisce da v15 solo in 26 celle onset (10 finestre di validation, 5 di private; nessuna ongoing). Estraendo dalla CV le differenze delle finestre cambiate (media +0,14, 30% negative), una somma ≤ −1,3 su 10 finestre esce con probabilità 0,5%. Causa non trovata. Per regola il Task 2 non si sceglie sulla classifica, e ogni misura locale (fold, mondi spostati, ufficiali) dice stack; la classifica dice il contrario fuori dal rumore. Decisione: v15 resta il riferimento, v16 è candidato per uno dei due invii finali.

**E1 per il Task 2** (`task2_modello.scarto_e1`, `cv_task2.costruisci_righe(e1=True)`, `cv_task2_stack.py mondo_righe/mondo --e1`). Per ogni finestra: mediana delle velocità osservate ed eleggibili dello strato mascherato nelle notti (1:00–4:00) del mese fino a T compreso, meno il livello notturno del train; lo scarto si toglie dalle velocità di storia e di T con il peso del flusso libero (pieno sopra 0,95·v_f, nullo sotto 0,8·v_f), poi caratteristiche e modelli invariati. Nessun dato dopo T (verifica online 48/48). Nei mondi spostati l'errore della stima è 0,09 km/h (0,24 il primo giorno del mese) contro uno scarto vero di 1,26; nelle finestre onset di validation e private lo scarto stimato ha deviazione standard 1,4 km/h.

| Onset attuali, Δ di E1 [bootstrap 95%] | Mondo del train | Mondo spostato 0 | Mondo spostato 1 |
|---|---|---|---|
| LightGBM | −0,0007 [−0,0025; +0,0012] | +0,0120 [+0,0077; +0,0166] (3/3) | +0,0067 [+0,0021; +0,0109] (2/3) |
| stack dei tre motori | +0,0009 [−0,0016; +0,0035] | +0,0033 [−0,0010; +0,0079] | −0,0017 [−0,0062; +0,0032] |

Con E1 LightGBM vale 0,8643 in tutti e tre i mondi: lo spostamento sparisce. Lo stack non ne ha bisogno. Sulle 40 ufficiali LightGBM con E1 fa 0,8982 contro 0,9169 nel mondo del train (una finestra), +0,013 e −0,027 nei mondi spostati: 40 finestre non dicono nulla. Diventa v17 per l'onset; le ongoing non sono ancora provate.

**Ongoing** (stessa prova, ricetta di v15: attuali più casuali dei semi 0–5, 26 milioni di righe; valutazione sulle casuali dei semi 0–2, 9.752 finestre, soglia 0,45):

| Metodo | IoU | Δ su LightGBM [bootstrap 95%] | Fold peggiore | Attuali (2.094) | Ufficiali (40) |
|---|---:|---|---:|---:|---:|
| LightGBM (v15) | 0,8423 | — | 0,8359 | 0,8900 | 0,8586 |
| XGBoost | 0,8527 | +0,0104 [+0,0089; +0,0122] | 0,8464 | 0,9117 | 0,8665 |
| CatBoost | 0,8372 | −0,0051 [−0,0069; −0,0033] | 0,8316 | 0,8801 | 0,8539 |
| media LightGBM + XGBoost | 0,8491 | +0,0069 [+0,0058; +0,0081] | 0,8426 | 0,9004 | 0,8662 |
| stack: caratteristiche + p di LightGBM | 0,8522 | +0,0100 [+0,0082; +0,0116] | 0,8470 | 0,9048 | 0,8570 |
| stack con contesto | 0,8606 | +0,0184 [+0,0160; +0,0209] | 0,8555 | 0,9076 | 0,8610 |
| **stack con le p dei tre motori e contesto** | **0,8650** | **+0,0228 [+0,0201; +0,0255]** | **0,8586** | **0,9106** | **0,8637** |

- Il contrario dell'onset: qui XGBoost (profondità 8, fino a 256 foglie) batte LightGBM (63 foglie) e CatBoost perde. Con 13 milioni di righe per fold conta probabilmente la capacità.
- Lo stack dei tre motori batte lo stack con il solo contesto di +0,0044 [+0,0031; +0,0059] e XGBoost da solo di +0,0123 [+0,0100; +0,0146], 3 fold su 3: per la regola vince. Soglia sul plateau (0,8642–0,8650 tra 0,40 e 0,50).
- Etichette lisciate (`scripts/cv_task2_etichette.py`: coda = velocità osservata mediata su 3 o 5 passi centrati ≤ 0,6·v_f, per avvicinare lo stato sottostante della verità ufficiale): cambia l'etichetta del 2,4% delle celle in coda; stesso ordine dei metodi (stack dei tre motori +0,0207 e +0,0188, XGBoost +0,0091 e +0,0074, intervalli sopra lo zero). Il rumore delle etichette non spiega i mancati trasferimenti delle ongoing.
- Mondo spostato (`mondo_lgb`, casuali del seme 0): LightGBM 0,8400 → 0,8396, XGBoost 0,8519 → 0,8516; con E1 0,8400 e 0,8517. Le ongoing non sentono lo spostamento del flusso libero (conta la dinamica della coda): E1 non serve. Lo stack non è stato provato nel mondo spostato, perché i suoi ingressi di primo livello non cambiano.
- Diventa v18: v17 con lo stack dei tre motori sulle ongoing (`allena_task2_stack.py --condizione ongoing`).
- Esito di v18 (28/10): 0,85998 contro 0,85624 di v17, +0,0037 (atteso ≈ 0,12 × 0,023 ≈ +0,0027). Le ongoing si trasferiscono, l'onset dello stack (v16) no: per i due invii finali, v18 e v18 con l'onset dello stack.

### Task 4: cosa dicono i dati (19/10/2026)

- I conteggi stanno solo sui tratti principali e sono coerenti con la struttura dei percorsi: la NNLS ufficiale li riproduce al 100% (S_link 1,0000). Il rango dei tratti contati è il numero di zone meno uno: i conteggi dicono quanti veicoli attraversano ogni sezione, non da dove a dove.
- Il prior ha troppi viaggi lunghi: A·b supera i conteggi di 1,5–2,5 volte (tranne D12_I405). Per adattarsi, la NNLS azzera il 50–66% dei percorsi, quasi tutti lunghi.
- Variante `kl_prior`: minima entropia relativa rispetto al prior con i conteggi come vincolo (f = b·exp(Aᵀμ)). Ogni percorso viene corretto dal prodotto dei fattori dei tratti attraversati, quindi una distorsione sulla lunghezza dei viaggi si corregge senza azzerare nulla. Anche questa riproduce i conteggi al 99,99%; il rapporto f/b scende con la lunghezza del percorso in modo regolare (da 0,85 a 0,06 per quintile su D7_I10_E).
- S_od, S_dev e S_attr richiedono i flussi di riferimento, che non sono rilasciati nemmeno per il train. Tra le due soluzioni, entrambe esatte sui conteggi, decide la classifica: un solo confronto tra due stimatori classici, con l'effetto sistematico su tutti i 10 corridoi e il prior costruito allo stesso modo in validation e private (rapporti conteggi/prior stabili tra gli split entro il 5%).
- I conteggi sono sintetici: sui tratti con lo stesso insieme di percorsi coincidono esattamente, quindi sono stati generati come A·f. Rispetto ai flussi osservati 15–19 la mediana del rapporto è 1,00, ma il singolo tratto va da 0,65 a 1,2.
- I tre prior (train, validation, private) hanno una parte comune e una propria di ogni split: corr(log b_train, log b_val) = 0,73–0,75 su tutti i corridoi, con dispersione del log 1,2 e una parte propria di σ ≈ 0,62, uguale ovunque. La correlazione 0,50 tra le differenze (log b_val − log b_train, log b_priv − log b_train) invece non prova nulla: viene fuori per qualunque terna di rumori indipendenti con la stessa varianza (revisione del 19/10). Asimmetria e curtosi dei residui sono compatibili con un moltiplicatore Gamma(3), ma non lo dimostrano. I conteggi cambiano tra i mesi del 3% circa.
- Esito in classifica (19/10): la KL perde circa 0,15 di S_ODME, la NNLS sulla media dei tre prior circa 0,26. Sappiamo solo che, sul mese pubblico e a parità degli altri compiti, queste due alternative rendono peggio della NNLS sul prior dello split; non che il riferimento nascosto sia una soluzione NNLS. Dopo due prove bocciate con margine netto, il Task 4 resta la NNLS ufficiale (S_ODME pubblicato 0,836) e non si cercano λ o varianti sulla classifica.

## Revisioni esterne: cosa è servito (20/10/2026)

Due documenti di altri agenti, scritti il 19/10 dopo v6 leggendo solo file e fonti: `REVISIONE_AVVERSARIALE.md` e `OSSERVAZIONI.md`. Giudizio: utili per validazione e riproducibilità, nessun guadagno di classifica diretto. La causa che ha dato v10 (+0,0035) non l'hanno vista: la revisione cita la «protezione delle finestre Task 2» tra i motivi per diffidare dei confronti di distribuzione (§10), senza arrivare ai buchi.

Hanno cambiato qualcosa:

| Proposta | Esito |
|---|---|
| Versioni congelate con manifesto e hash, ID controllati contro `sample_submission.csv` (revisione §12–13) | Fatto (`genera_invio.py`, `invio.py`). v8 e v10 si rigenerano identici dalla loro cartella; serve anche alla riproducibilità entro l'1% |
| Stima fisica allineata al valutatore: transizioni, rampe, S_FD (revisione §6–7, osservazioni 3.1) | Fatto (`s_fisica_stima`, `s_fd_valutato`). S_LWR di v1 cambia al più di 0,002: nessuna decisione precedente cambia. Ha escluso la stima come causa di v7 e v9 e reso credibile quella di v10 (+0,004 attesi, +0,0035 osservati) |
| Valutare sulle 80 finestre ufficiali del train (osservazioni R1 e §5.4, revisione §3) | Ha mostrato che le ongoing ufficiali sono più difficili delle ricostruite e scelte quasi a caso: da qui v8 (0,835 → 0,857 fuori campione, in classifica −0,0013, dentro il rumore del Task 2). Spiega in parte lo scarto v1 → v3 (+0,116 contro +0,144 attesi) |
| La correlazione 0,50 tra le differenze dei prior non prova nulla (revisione §11) | Giusto: tolta la deduzione dal README |

Controlli fatti che hanno solo confermato: frequenze di coda calcolate senza il fold valutato (±0,004, nessun ottimismo), origine progressiva (con 6 mesi si ritrova la stima a blocchi), Task 2 senza frequenza (serve: onset −0,034), S_FD esatto per le versioni inviate (non spiega v7 e v9). Si veda «Task 2: controlli della revisione».

Dove erano deboli:
- Il margine più grande indicato era un post-processing fisico di v e q (osservazioni §5.1). Il guadagno fisico vero è venuto dal correggere una differenza tra train e mesi valutati, non da un vincolo fisico.
- Le osservazioni davano le versioni già congelate in un manifesto (§1.4); allora non lo erano, e la revisione (§12) aveva ragione.
- I «0,5–1 punti di S_state» attesi dalla media di più seed (osservazioni R3) sono una stima non misurata.
- L'avvertimento sulle scelte ripetute sulla stessa CV (osservazioni 3.7) è stato confermato da v7 (3.200 alberi, non trasferito), anche se la causa era un'altra.

Non ancora fatti: media di più seed (R3), pesi per famiglia o corridoio (R4, revisione §5), `deterministic` in LightGBM (R5; il seed è fissato), post-processing fisico (§5.1), struttura temporale del Task 2 (§5.2), distanze fisiche tra i tratti (revisione §9), report (R7). Determinismo e report entrano comunque nel congelamento finale.
