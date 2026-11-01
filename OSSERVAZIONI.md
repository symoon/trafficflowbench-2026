# Osservazioni e proposte di miglioramento

Documento esterno al codice (non modifica nulla). Basato sulla lettura di `README.md`,
`docs/`, `src/tfb/`, `scripts/`, `ufficiale/docs/` e degli output locali allo stato del
19/10/2026.

Punto di partenza: **v6** in classifica pubblica a **0,85232** (209°/513), contro 0,924
del 1° e 0,9945 della risposta perfetta. Metodo: Task 1 LightGBM sui residui
dell'interpolazione, Task 2 LightGBM per condizione, Task 4 NNLS ufficiale.

Scadenze: registrazione entro 25/10, fine invii 07/11, pacchetto + report 10/11.
Il report pesa il 15% del voto finale.

---

## 1. Cosa funziona e va tenuto

L'impostazione di metodo è tra le più solite che si possano avere in una competizione
di questo tipo. Non la toccherei:

1. **Principio «niente overfitting»** scritto e rispettato: le scelte si fanno sulla
   validazione a blocchi del train, la classifica serve solo a controllare il trasferimento.
   Le bocciature nette di v4 e v5 (KL, prior medio) dimostrano che la regola viene applicata.
2. **Indagine fisica prima dei modelli.** Rumore irreducibile, correlazione dei flussi
   tra tratti, rampe scollegate, densità = q/v, analisi dei prior: tutto ciò guida le scelte
   e risparmia esperimenti inutili (la densità diretta è stata scartata con un +0,0003).
3. **Conformità al vincolo online del Task 2** verificata con `verifica_task2_online.py`
   su 48 finestre, e documentata in `docs/conformita.md`.
4. **Riproducibilità**: `genera_invio.py` congela modelli, intermedi, hash e versioni in un
   manifesto. È più strutturato di quanto serve per la gara e va benissimo così — anzi,
   è un requisito di premiazione (ripetere entro l'1% del privato).
5. **Caratteristiche del Task 1** ben pensate: residuo sulla retta del buco, forma dei
   vicini attraverso lo stesso buco, densità del contesto. Sono «model-free» e si
   trasferiscono tra corridoi.
6. **Abbandono onesto** delle varianti che perdono, senza tarare sulla classifica.

---

## 2. Stima di dove sono i punti che mancano

Pesi: `0,35·S_state + 0,30·S_queue + 0,15·S_physics + 0,20·S_ODME`.
Ordini di grandezza da validazione locale e punteggi pubblici noti:

| Termine | Stima attuale | Peso | Tetto pratico | Margine sul totale |
|---|---:|---:|---:|---:|
| S_state | ~0,94 locale (fold LGBM) | 0,35 | 1,00 | ~0,02 |
| S_queue | ~0,89 (0,865 onset / 0,914 ongoing) | 0,30 | 1,00 | ~0,03 |
| S_physics | ~0,60–0,70 stimato; 0,96 il tetto ufficiale | 0,15 | 0,96 | ~0,04–0,05 |
| S_ODME | 0,836 (NNLS ufficiale, pubblico) | 0,20 | 1,00 | ~0,03 |

Conti fatti: 0,35·0,94 + 0,30·0,89 + 0,15·0,65 + 0,20·0,836 ≈ 0,853, in linea con v6.
Per avvicinarsi al 0,924 servono miglioramenti **su più fronti**, non un solo termine.
Il Task 3 e il Task 2 sono i due con più margine *e* con più leva; il Task 4 ha margine
ma l'hanno già esplorato senza successo.

---

## 3. Debolezze dell'approccio attuale

### 3.1 Stima locale di S_LWR solo parziale

`s_lwr_stima` usa le variazioni vere di N = (q/v)·L sulle sole transizioni temporali.
Il valutatore ufficiale usa flussi di confine degli organizzatori (proiettati sulle
osservazioni), le rampe e le modalità di copertura (tutti i corridoi sono Mode A).
Quindi:

- la stima **non è il punteggio** e potrebbe ordinare male due varianti vicine;
- non c'è un modo locale di sapere se l'errore si concentra sui termini di bordo,
  sulle rampe o sui passi temporali interni.

Finora l'ordine delle varianti (v6 meglio dell'interpolazione, la densità quasi neutra)
è sembrato coerente con la classifica, ma ogni scelta futura che muove solo il Task 3
dovrebbe trattare la stima come un **segnale debole**, non come metrica.

### 3.2 Task 2: addestramento su finestre ricostruite, non su quelle ufficiali

`finestre.costruisci` imita la regola di selezione e produce 4.196 finestre, ma ne
recupera solo 47–53 delle 80 ufficiali del train. Le finestre ufficiali **ci sono**
(`window_index.csv`, 5 onset + 5 ongoing per corridoio per split) e portano
`history_coverage` e `future_coverage` veri.

Conseguenze:

- la CV a blocchi misura la capacità sulle finestre ricostruite, non su quelle che il
  valutatore sceglierebbe;
- le soglie 0,40 / 0,45 sono scelte su quella distribuzione;
- la differenza può non essere grande (le regole sono quasi le stesse), ma non è stata
  quantificata.

### 3.3 Sbilanciamento tra corridoi nell'obiettivo

S_total fa media **per famiglia** (5 famiglie, direzioni uguali). LightGBM è un modello
unico allenato con MSE sulle celle, quindi pesa i corridoi con più celle bersaglio.
Un miglioramento su un corridoio «grande» può non muovere il punteggio quanto uno su
una famiglia sottorappresentata. Non è un errore: è un disallineamento tra loss e metrica.

### 3.4 La frequenza storica del Task 2 può non trasferire

La feature `frequenza` (quota di giorni in coda per ora e tratto) è probabilmente tra le
più forti per onset e ongoing. Ma la privata è generata **con altri incidenti**: se i
colli di bottiglia si spostano, la storia del train diventa un prior sbagliato.
Le caratteristiche geometriche (distanza dalla coda, r = v/v_f, pendenza) sono più
robuste; la frequenza lo è meno. Finora non c'è un ablation che lo misuri.

### 3.5 Previsione per passi indipendenti

Per `queue_ongoing` si allena un solo modello con feature `passo` e si predicono i 6
passi in modo quasi indipendente. L'evoluzione di una coda (propagazione a monte alla
velocità d'onda, ritiro) ha struttura temporale: due passi adiacenti non sono casi
separati. Nessun vincolo di monotonia o di continuità spaziale è imposto.

### 3.6 Onset: si usa solo T+30, e va bene, ma con poco segnale

Per `queue_onset` si prevede solo T+30 (confermato dagli organizzatori). Persistenza
vale 0,002: il modello deve indovinare **quali link** si accoderano tra mezz'ora.
È il caso più difficile del Task 2 e ha il fold peggiore più vicino alla soglia.
Le feature a 30 minuti (frequenza, geometria attuale) sono tutto ciò che si ha.

### 3.7 Selezione multipla sulla stessa CV (stesso tema di Name That LLM)

Alberi del Task 1 (200/400/800), soglie del Task 2, varianti di caratteristiche: tutto
guardando i tre blocchi. I margini tra lgb_400 e lgb_800 sono di qualche millesimo,
ma la scelta finale tra pipeline vicine (e la scelta degli invii finali) risente dello
stesso effetto. Non è un problema per v6 vs v1; lo è se si inizia a scegliere tra 5–6
varianti simili.

### 3.8 Task 4 bloccato, ma non spremuto

La NNLS ufficiale sta a 0,836. Due alternative di principio (KL, prior medio) hanno
perso 0,15 e 0,26 di S_ODME: segnale chiaro che la metrica è agganciata al prior
rilasciato dello split e a soluzioni sparse. Non forzare. Restano però varianti **non
ancora provate** e coerenti con quel segnale (vedi §4.5).

### 3.9 Un solo seed, multi-thread

`conformita.md` lo sa già: LightGBM multi-thread non è deterministico. Per la
riproduzione entro l'1% va fissato `deterministic=true` e verificato un rilancio da
zero. Meglio prima che dopo.

---

## 4. Miglioramenti rapidi (poco rischio, alto ritorno atteso)

In ordine di ritorno atteso per sforzo.

| # | Intervento | Perché | Come verificarlo |
|---|---|---|---|
| R1 | **CV su finestre ufficiali del train** (80) accanto a quelle ricostruite | Misura lo scarto di distribuzione; regola le soglie su ciò che conta | Valutare le predizioni già salvate sulle finestre ufficiali; se lo scarto è grande, ri-tarare lì |
| R2 | **Ablazione `frequenza`** (senza, con rumore, con solo le prime settimane) | Capisce quanto il Task 2 dipende da un prior che la privata può spezzare | Rerun di `cv_task2` senza la feature |
| R3 | **Media di 3–5 seed** (o bagging) di LightGBM per Task 1 e Task 2 | Stabilizza; guadagni tipici 0,5–1 punto di S_state locale | `cv_task1` con seed diversi, media delle predizioni |
| R4 | **Loss / pesi per famiglia** (o per corridoio) nel Task 1 | Allinea l'obiettivo alla metrica a media di famiglia | Confronto appaiato per famiglia su `cv_blocchi` |
| R5 | **Deterministic LightGBM + rilancio da zero** | Requisito di premiazione, non un optional | Due run identiche → stesso CSV |
| R6 | **Riconciliazione v–q sul diagramma fondamentale** come post-processo leggero | S_LWR pesa 2/3 di S_physics; proiettare (v,q) verso il triangolo con vincolo debole costa poco | Misurare S_state *e* stima S_LWR prima/dopo; accettare solo se S_state non scende |
| R7 | **Report** (15% del voto) abbozzato adesso | È l'unica cosa che non migliora con gli invii | Struttura, figura dei 4 task, tempi, hardware |

---

## 5. Esperimenti a più alto valore (medio rischio / tempo)

### 5.1 Post-processing fisico del Task 1 (il margine più grande)

Oggi v e q sono predetti come residui indipendenti e poi clippati. Il punteggio di
fisica guarda le variazioni di N = q/v e la coerenza con il triangolo. Varianti da
confrontare **sulla stessa CV**, con la stima di S_LWR e senza toccare S_state:

1. proiezione soft sul vincolo di conservazione tra celle consecutive (con le rampe
   osservate), regolarizzata per non allontanarsi dalla predizione LGBM;
2. predizione congiunta di (v, k) con q = k·v, così N è lineare in k;
3. ottimizzazione locale di (v,q) che minimizzi `α·|Δ(v,q)|² + β·|ΔN̂ − ΔN_stima|²`
   nelle sole celle oscurate.

La densità diretta ha guadagnato poco; qui non si sostituisce il modello, si
**corregge** la coppia predetta. È la strada più promettente per il +0,02–0,04 di
S_physics.

### 5.2 Task 2 con struttura temporale e spaziale

- **Propagazione alla velocità d'onda**: feature (o post-processo) che estenda la coda
  a monte di `v_w·Δt`, con `v_w` dal diagramma fondamentale del tratto.
- **Sequenza**: un piccolo modello (o LightGBM con feature del passo −1/−2 predetto)
  che vincoli la geometria della coda a non spezzarsi senza motivo.
- **Onset a 30 minuti**: feature più dirette di «cosa succede tipicamente a quest'ora
  su questo tratto» e di quanto la coda attuale sia *in ritardo* rispetto alla sua
  storia. Il guadagno possibile è concentrato qui.
- **Soglie per finestra** (in base a quanti link sono già in coda) invece che per sola
  condizione: le curve locali sono piatte, ma il comportamento ottimo di onset e ongoing
  è diverso quando la coda è piccola o grande.

### 5.3 Un secondo Task 1 «di contrasto» per il blend locale

Interpolazione (v1) e LGBM (v6) sbagliano in modo diverso: la media pesata delle
predizioni, con peso scelto a blocchi, spesso batte il singolo modello. Costo basso
(una media), da provare prima di modelli più complessi. Vale per v e q.

### 5.4 Dati «quasi ufficiali» per il Task 2

Se il recupero delle 80 finestre ufficiali resta incompleto, si può:

- allineare meglio la regola (distanza 360 minuti per condizione vs unica, filtro
  IoU della persistenza, copertura ≥ 0,7) fino a recuperarne di più;
- oppure generare *più* finestre con la stessa regola ma **meno filtri aggressivi**,
  e pesare di più quelle che assomigliano alle ufficiali.

### 5.5 Task 4: varianti coerenti con il segnale ricevuto

La classifica ha detto: tenere il prior dello split e una soluzione con zeri (NNLS).
Varianti che non contraddicono:

- **λ piccolo e scelto sui soli conteggi** (fit di A·f ai conteggi, non alla metrica
  nascosta): se S_link è già 1,0000, λ muove solo S_dev/S_attr.
- **NNLS + scarto quadratico debole sui percorsi lunghi** (il prior ha troppi viaggi
  lunghi; la NNLS li azzera — forse un po' meno che azzera, un po' più che KL).
- **Niente parametri scelti sulla classifica pubblica**: due bocciature sono già un
  campione piccolo e rumoroso (S_od non è osservabile in locale).

Attesa realistica: qualche millesimo, non 0,05. Da fare solo dopo R1–R6 e il §5.1.

### 5.6 Modello più ricco per il Task 1 (solo se i gradini di sopra reggono)

Gradient boosting più profondo, reti piccole sulle serie temporali del tratto e dei
vicini, o un modello a fisica ibrido (LWR discreto con correzione data-driven).
La GPU del server 192.168.0.3 è disponibile. Ma i dati sono sintetici e già
ben spiegati da un LGBM a 48 caratteristiche: prima di scalare, esaurire le leve
strutturali (§4 e §5.1–5.3).

---

## 6. Approccio alternativo (se si ricomincia o come secondo modello)

**Simulatore di stato + apprendimento del residuo.**
Partire da un integratore LWR sui parametri pubblici del diagramma fondamentale
(topologia, rampe, capacità), alimentato dai flussi osservati ai bordi delle celle
oscurate; poi correggere il solo scarto tra simulazione e osservazioni con un modello
dati-driven. Pro: S_LWR e S_FD quasi costruiti; il modello impara solo ciò che la
fisica non sa (rumore, incidenti, comportamento delle rampe). Contro: il valutatore
usa flussi di confine *propri* e proiettati — un simulatore che non li conosce può
discostarsi; serve una validazione locale che tenga conto di questa asimmetria.

È la direzione giusta se il §5.1 (correzione leggera) dà segnale ma non abbastanza.
Non la sostituirei al pipeline attuale per questa gara: tempo breve e 0,852 è già
una base solida.

---

## 7. Metodologia di validazione (da rinforzare)

1. **Tre livelli di prova**, nell'ordine: CV a blocchi sulle finestre/celle ricostruite,
   CV sulle finestre ufficiali del train, controllo di coerenza con la classifica
   (solo come termometro, mai per scegliere i parametri).
2. **Confronti appaiati per corridoio e per fold**, non solo le medie. Un +0,002 medio
   che si concentra in una famiglia non vale quanto un +0,001 su tutte e cinque.
3. **Metriche di robustezza**: fold peggiore, famiglia peggiore, e — per il Task 2 —
   IoU per finestra, non solo la media (la deviazione standard dichiarata è 0,375).
4. **Niente scelte sulla pubblica** per Task 1, 2 e (dopo le due bocciature) Task 4.
   Se serve un feedback sul Task 3, un solo invio di controllo, già previsto nel piano.
5. **Stima di S_LWR trattata come proxy**: ogni volta che due varianti sono entro
   lo scarto della stima, preferire la più semplice.

---

## 8. Strategia d'invio e pacchetto

- **Diversificare gli invii finali**: es. v6 (attuale) + una variante con post-processing
  fisico o blend interpolazione/LGBM. Due varianti troppo simili cadono insieme.
- **Congelare** la versione scelta con `genera_invio.py` e verificarne il rilancio da
  zero, cache comprese (`data/cache` va ricostruita, non riusata).
- **Tempi e memoria** del rilancio completo nel report: serve a dimostrare la ripetibilità
  entro l'1%.
- **Registrazione entro il 25/10** (tra sei giorni): è il vincolo più vicino e non ha
  a che fare con i modelli. Verificare che username e membri coincidano con la squadra
  Kaggle.
- Il **report PDF** merita lavoro vero: 15% del voto, e le squadre che lo trascurano
  escono dalla rosa dei premi anche con un buon punteggio.

---

## 9. Priorità proposte

Con ~2,5 settimane e 5 invii al giorno:

1. **Registrazione (25/10)** e **R5** (determinismo) — prerequisiti, non negoziabili.
2. **R1 + R2** — sapere se il Task 2 regge sulle finestre ufficiali e quanto dipende
   dalla frequenza storica. Guadagna informazione prima di investire.
3. **§5.1 post-processing fisico** — il margine più grande sul totale (S_physics).
4. **R3 + R4 + §5.3** — stabilizzazione e loss allineata alla metrica: poco rischio,
   guadagni piccoli ma solidi.
5. **§5.2** struttura del Task 2, in particolare onset.
6. **R7 report** in parallelo, aggiornato a ogni invio serio.
7. **§5.5 Task 4** solo se resta tempo: attese piccole.
8. **§5.6 modelli più ricchi** solo se 3–5 hanno dato e resta tempo/GPU.

Da evitare: tarare soglie o λ sulla pubblica; varianti del Task 4 non giustificate dai
conteggi; modelli grandi prima di aver chiuso i buchi di validazione (R1, R2); due
invii finali che condividono le stesse debolezze.

---

## 10. Note di implementazione (se si seguono le proposte)

- R1–R6 e §5.1–5.3 stanno nel perimetro attuale (`scripts/cv_*.py`, `src/tfb/`):
  nessun cambio di formato d'invio, nessuna nuova dipendenza pesante.
- Le finestre ufficiali del train si leggono già con `tfb.task2.finestre(p, "train")`;
  serve uno script di valutazione parallela, non un cambio di dati.
- Il post-processing fisico (§5.1) va valutato **insieme** a `s_state` e `s_lwr_stima`:
  un guadagno di stima con perdita di S_state non entra.
- Per i seed multipli: salvare le predizioni per seed, mediare in probabilità (Task 2)
  e in spazio (v, q), poi riusare `genera_invio.py` senza cambiare la struttura.
- Ogni nuovo file di output andrebbe messo sotto `outputs/` con la versione nel nome,
  come già fatto per gli invii: il manifesto regge solo se gli hash sono unici.
