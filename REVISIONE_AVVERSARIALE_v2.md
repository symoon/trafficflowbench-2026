# TrafficFlowBench — revisione avversariale v2

20 ottobre 2026. Basata sul README aggiornato, sul codice e sugli artefatti locali; ricerca di metodi complementari su pubblicazioni originali e repository degli autori.

**Direzione proposta:** passare dalla correzione indipendente delle celle a una ricostruzione condizionata dalla geometria dei mancanti e, nei buchi difficili, dell’intera traiettoria. Sul Task 2 proverei una decisione orientata all’IoU e alla struttura temporale delle code prima di aumentare la capacità del classificatore. La normalizzazione del livello del mese merita una prova diversa dall’aggiunta di tre caratteristiche, già in lavorazione.

L’unico file scritto per questa revisione è questo documento. Non ho modificato codice, dati, modelli, cache, invii o altri documenti; non ho eseguito test, training, analisi numeriche sui dati o comandi Kaggle, né usato il server GPU. I numeri provengono dal README e dagli artefatti letti; i metodi proposti non sono stati sperimentati qui.

## 1. La diagnosi è cambiata: partire da quello che ha funzionato

La prima revisione ha aiutato a verificare validazione, metrica e riproducibilità. Le criticità delle frequenze del Task 2 erano reali nella costruzione della CV, ma correggerle ha spostato i risultati soltanto di circa ±0,004. L’allineamento dei filtri fisici ha cambiato poco le stime. Non sono più le spiegazioni principali del mancato trasferimento.

Il risultato più informativo è **v10**:

| Confronto documentato | Risultato | Cosa suggerisce |
|---|---|---|
| v6 → v7, 800 → 3.200 alberi | CV migliore, pubblico 0,85232 → 0,85128 | La capacità aggiuntiva ha aiutato la distribuzione sbagliata |
| v8 → v9, randomizzazione del livello di velocità | Migliore nei mondi simulati, pubblico 0,85098 → 0,85050 | Il cambiamento simulato non copriva il problema dominante |
| v8 → v10, modello dedicato alla regione dei blackout | Pubblico **0,85098 → 0,85449** | Riprodurre la geometria reale dei mancanti ha prodotto un guadagno trasferibile |
| Task 2, training ongoing su unione di selezioni | 0,835 → 0,857 sulle 40 finestre ufficiali train fuori campione | La selezione degli episodi conta; il confronto pubblico resta poco informativo |

Fonti: [README](README.md), [simulazione dei buchi](scripts/cv_buchi.py), [selezione Task 2](scripts/cv_task2_selezione.py). I punteggi pubblici sono riportati dal progetto, non ricontrollati via API.

La lezione di v10 è più specifica di “serve più fisica”: **un insieme piccolo di celle con un contesto assente dal training può dominare gli errori sulle variazioni di densità**. Il nuovo ciclo di esperimenti deve cercare altri contesti di questo tipo e modellarne esplicitamente l’incertezza.

Non riproporrei come prima attività i controlli già conclusi. Né assumerei che l’insuccesso di v9 abbia dimostrato l’irrilevanza dello spostamento tra mesi: il README stesso lascia aperto il suo contributo fuori dai buchi.

## 2. Due limiti nuovi da chiudere nella valutazione del riferimento v10

### 2.1 La regione locale usa la verità fuori dalle predizioni sostituite

In `cv_buchi.py`, `misura` costruisce `Nh` copiando il numero vero di veicoli `N`, poi sostituisce soltanto i bersagli nella regione T…T+24. È una misura utile dell’effetto locale. Nell’invio, però, fuori dalla regione restano le predizioni del modello standard, non la verità.

In questa notazione del codice, +24 indica 24 passi da cinque minuti, cioè 120 minuti; il blackout T+1…T+18 copre invece da 5 a 90 minuti dopo l’origine.

Il residuo contiene differenze temporali e valori assoluti. Per una transizione al confine, l’errore esterno può sommarsi o compensare quello della regione modificata. La perdita non è additiva:

`|errore_esterno + correzione| − |errore_esterno|`

non coincide in generale con la perdita ottenuta ponendo `errore_esterno = 0`.

La scalatura da circa 25 finestre per blocco a dieci per mese è inoltre un’approssimazione del punteggio, che media regimi e pannelli e applica limiti a zero. Il fatto che abbia previsto bene v10 è un buon riscontro empirico, non una garanzia per ogni nuova correzione.

**Prossima misura:** pseudo-mesi completi del train con dieci finestre per pannello, storie protette e blackout riprodotti. Applicare v10 a tutti i bersagli; poi sostituire soltanto il ramo candidato, preservando altrove le sue effettive predizioni. Calcolare S_state, S_FD e il proxy LWR allineato nella loro aggregazione completa.

Conservare il controllo regionale come diagnostica, insieme al risultato sull’intero pseudo-mese. La differenza chiarirà se un candidato migliora il buco o semplicemente ne sposta l’errore sul bordo.

### 2.2 Rimascherare osservazioni non replica automaticamente i bersagli difficili

`rimaschera_valutati.py` nasconde una quota delle celle ancora osservate. Con una quota aggiuntiva del 10%, una maschera casuale originaria r diventa, nel caso ideale:

`r_effettivo = r + (1 − r) × 0,10`.

R1 passa quindi dal 20% al 28%, R2 dal 30% al 37%, R3 dal 50% al 55%. Il contesto cambia e il cambiamento non è uguale tra regimi. Soprattutto, dentro un blackout completo non ci sono osservazioni da scegliere: proprio quelle condizioni restano poco rappresentate. L’opzione `--vicino` campiona celle osservabili vicine ai buchi, non crea verità all’interno di essi.

**Interpretazione corretta:** il controllo misura trasferimento su osservazioni rilasciate rese artificialmente mancanti. È utile e usa dati consentiti per il Task 1; non dimostra da solo il miglioramento sulle celle nascoste dalla selezione del Task 2.

**Azione:** mantenere tre prove distinte: rimascheratura puntuale sui mesi rilasciati, blackout su train con verità disponibile, pseudo-mesi con la procedura completa. Un metodo deve spiegare dove guadagna e dove perde in ciascuna.

## 3. Un limite informativo che i modelli non possono eliminare

Nel buco del Task 2 sono oscurati per 90 minuti tutti i rilevatori della linea principale nelle celle pertinenti. Non è il classico caso di un singolo sensore mancante con i vicini osservati.

Una rete a grafo, una fattorizzazione o un Transformer non possono ricavare dai vicini contemporanei una misura che non esiste da nessuna parte del corridoio. Devono usare stato precedente, osservazioni successive consentite al Task 1, calendario, ripetizioni tra giorni e regolarità degli episodi.

Se l’inizio o la durata di un incidente è imprevedibile dal contesto rilasciato, rimane incertezza. Una curva perfettamente liscia può essere sbagliata quanto una previsione molto dettagliata. Per questo i benchmark di imputazione con mancanti casuali sono evidenza insufficiente per scegliere il metodo da inserire qui.

**Domanda da porre a ogni candidato:** quali osservazioni nuove sfrutta rispetto a v10, oppure quale dipendenza tra osservazioni esistenti riesce ad apprendere meglio?

## 4. Opportunità e priorità: sette esperimenti realmente diversi

| ID | Approccio | Limite che affronta | Prima prova | Costo iniziale |
|---|---|---|---|---|
| E1 | Ricostruzione equivalente agli scarti di livello e denoising del flusso libero | Memorizzazione del livello del tratto e interpolazione del rumore | Livello mensile reinserito esplicitamente nell’uscita | CPU |
| E2 | Ricostruzione di sequenze condizionata dalla maschera | Correzioni indipendenti in blackout lunghi | Piccola rete temporale sui soli episodi | GPU, modello piccolo |
| E3 | Obiettivo congiunto su v, q e variazioni dell’errore di densità | Basso RMSE senza coerenza tra istanti consecutivi | Stesso modello E2, due loss a confronto | Costo marginale dopo E2 |
| E4 | Interpolazione lungo direzioni di propagazione | Vicini scelti per indice e istante, non per dinamica | Adaptive smoothing come nuova base o caratteristica | CPU |
| E5 | Fattori temporali a basso rango, con adattamento al mese | Prior storico troppo rigido e forte ridondanza dei flussi | Modello piccolo dei residui di flusso | CPU; memoria da limitare |
| E6 | Decisione Task 2 per ranking e IoU della finestra | Un’unica soglia per eventi di dimensioni diverse | Nuovo decoder sulle probabilità OOF | CPU, senza nuovo backbone |
| E7 | Durata e propagazione coerenti delle code | Decisioni separate per tratto e orizzonte | Correzione temporale morbida o modello di fronte | CPU o piccola rete |

Priorità consigliata: valutazione completa del riferimento; **E6 ed E1** come prove contenute; poi **E2 + E3** come investimento maggiore. E4 ed E5 sono rami complementari da tenere soltanto se gli errori differiscono da LightGBM.

Non attribuisco a questi esperimenti un guadagno previsto in classifica. Le pubblicazioni citate dimostrano i rispettivi meccanismi su altri dati, non il beneficio su TrafficFlowBench.

## 5. E1 — Togliere davvero il livello del mese, poi restituirlo nell’uscita

Il progetto sta già provando `v_lib_mese`, `v_lib_scarto` e `v_it_meno_lib`. **Questa attività è in corso e non è una nuova proposta di questa revisione.** La distinzione utile è tra fornire al modello un’altra colonna e costruire un metodo che rispetti il cambiamento di livello per definizione.

### Versione alternativa da confrontare

Per ogni tratto e mese, stimare il livello di flusso libero `μ` dalle sole osservazioni rilasciate, con una stima robusta e un ripiego quando il campione è insufficiente. Poi:

- trasformare le velocità di ingresso in scarti rispetto a μ prima di costruire le caratteristiche;
- imparare un target relativo, per esempio `v_vera − μ`;
- ricostruire `v_predetta = μ + residuo_predetto`.

Un offset comune alle velocità del tratto cambia così il livello reinserito, senza chiedere agli alberi di imparare nuovamente la correzione. Occorre rendere coerenti anche vicini e riferimenti statici: lasciare tutte le velocità assolute tra gli ingressi può reintrodurre la scorciatoia.

Il risultato deve essere confrontato con l’aggiunta delle tre caratteristiche attuali e con il modello invariato, non con una baseline diversa. Separare la prova sui giorni normali da quella sui blackout; lo script finale corrente non combina ancora `--mese` e `--buchi`.

### Denoising condizionato al regime

Il README riporta un modello empirico molto forte per il flusso libero: livello del mese per tratto più rumore quasi bianco. In tale condizione, interpolare tra due osservazioni può introdurre il rumore degli estremi senza poter prevedere quello della cella mancante.

Come controllo teorico, se il livello fosse noto, i rumori fossero indipendenti con varianza σ² e si interpolasse a metà tra due estremi, l’errore quadratico atteso rispetto al target rumoroso sarebbe `1,5σ²`; prevedere il livello darebbe `σ²`. Sono ipotesi, non una stima del guadagno globale.

**Prova concreta:** confrontare una combinazione tra livello mensile e previsione v10, attivata dalla probabilità di essere in flusso libero e dalla qualità del contesto. Conservare il modello dinamico nelle transizioni e in coda. La probabilità deve usare grandezze disponibili, non la velocità vera.

Il riferimento metodologico è la normalizzazione reversibile di RevIN, concepita per separare cambiamenti di distribuzione e dinamica della serie. Il nostro adattamento deve gestire maschere e stimare statistiche soltanto dalle osservazioni consentite. [RevIN, ICLR 2022](https://openreview.net/pdf?id=cGDAkQo1C0p), [codice degli autori](https://github.com/ts-kim/RevIN).

La stima del mese intero è consentita per il Task 1 offline. Non va passata al Task 2: per quel compito qualsiasi normalizzazione del mese valutato deve fermarsi alla T della finestra. Inoltre non si modifica `v_f` ufficiale né la soglia delle etichette del Task 2 soltanto perché è cambiato il livello empirico.

## 6. E2 — Ricostruire il blackout come una sequenza, non come tante righe

Il modello v10 è addestrato sulle maschere giuste, ma continua a prevedere residui per cella, con due modelli distinti. Una rete temporale piccola può vedere insieme ingresso nel buco, uscita, andamento dei vicini e maschere, producendo un’intera regione.

### Prova minima

Usare una rete convoluzionale temporale o un piccolo encoder di sequenze su patch che includano almeno la storia protetta, i 90 minuti oscurati e un margine successivo. Ingressi:

- velocità relative al livello del tratto, flussi per corsia e occupazione;
- maschera di osservabilità di ciascun canale, separata dalla maschera dei bersagli da consegnare;
- distanza dall’ultima e dalla prossima osservazione;
- calendario e coordinate fisiche, limitando gli identificatori che consentono di memorizzare il tratto;
- un valore di partenza v10 o interpolato, mantenuto distinto dalle misure reali.

Uscite condivise per v e q, inizialmente come residui. La rete deve sostituire soltanto i bersagli richiesti, mantenendo immutate le osservazioni rilasciate. Un riempimento iniziale non deve diventare una falsa osservazione.

Il training deve riprodurre storie protette, blackout e maschere dei regimi, con gli episodi interi nello stesso fold. Le copie di finestre della stessa giornata non sono esempi indipendenti. Aggiungerle può ampliare il supporto delle maschere, non giustifica un intervallo di confidenza più stretto.

### Integrazioni pertinenti

**SAITS** usa attenzione e informazione dei mancanti per ricostruire serie multivariate. Offre un punto di partenza più mirato di un Transformer generico. [SAITS, ESWA 2023](https://arxiv.org/abs/2202.08516), [implementazione ufficiale](https://github.com/WenjieDu/SAITS).

**GRIN** apprende rappresentazioni spazio-temporali per imputazione tramite message passing. È interessante se le relazioni spaziali aggiungono segnale rispetto alla sola rete temporale. Qui occorre costruire un grafo coerente con ordine e distanze dei sensori, distinguendolo dalla topologia del valutatore, che include link non osservati e connettori. [GRIN, ICLR 2022](https://arxiv.org/abs/2108.00298), [codice ufficiale](https://github.com/Graph-Machine-Learning-Group/grin).

Non proverei entrambi subito: prima una rete temporale piccola, poi una componente spaziale se l’ablazione indica valore. **La bidirezionalità è ammessa solo nel ramo offline del Task 1.**

**Criterio di successo:** migliorare la procedura v10 sul pseudo-mese completo, senza limitarsi all’RMSE medio dentro il buco. Confrontare fronti di coda, fondo del blackout, uscita e transizioni ai bordi. Misurare anche tempo e picco di memoria; nessun benchmark del paper garantisce che il modello sia conveniente qui.

## 7. E3 — Imparare la coerenza delle variazioni, senza inventare flussi di confine

Il rapporto `q/v` rende la densità molto sensibile alla velocità bassa. Prevedere v e q con due loss quadratiche indipendenti non controlla direttamente gli errori su N o la loro differenza tra istanti.

La precedente prova di una densità diretta per cella ha prodotto un vantaggio piccolo. Il nuovo esperimento è diverso: **uscita condivisa di sequenza e loss sulle variazioni dell’errore**, applicata alle maschere reali.

Sul training, con verità disponibile, definire:

`e_N(t,l) = L_l · q_pred(t,l)/v_pred(t,l) − N_vero(t,l)`.

Aggiungere alla loss di ricostruzione:

`λ · media_pesata |e_N(t+1,l) − e_N(t,l)|`.

I pesi e le transizioni devono seguire l’universo ufficiale; includere i confini tra regioni modificate e predizioni del ramo standard. Si può confrontare anche una loss robusta su e_N, mantenendo piccola la griglia di λ.

**Distinzione essenziale:** penalizzare la differenza degli errori sul training non significa penalizzare tutte le variazioni di N. La seconda scelta cancellerebbe variazioni vere, comprese quelle degli incidenti e del rumore osservato che il valutatore usa come riferimento.

Questa loss non ricostruisce i flussi degli organizzatori: usa una proxy supervisionata disponibile nel train. Non suggerisco di imporre rigidamente la conservazione calcolata da rampe e topologia, dato quanto emerso sul loro rapporto con il segnale sintetico.

**Controlli:** stesso modello e stessi dati con/senza la loss; superficie di compromesso tra S_state e LWR, con S_FD riportato; performance nelle code e in flusso libero. Se si migliora soltanto LWR sacrificando più punteggio di stato di quanto si recuperi, fermarsi.

### Se emerge incertezza multimodale

Un incidente con tempi incerti può produrre traiettorie alternative: la loro media smussa i fronti. **CSDI** propone imputazione probabilistica mediante diffusione condizionata alle osservazioni. È una possibilità successiva, non il primo modello che installerei. [CSDI, NeurIPS 2021](https://arxiv.org/abs/2107.03502).

Campionare una traiettoria plausibile non garantisce un migliore RMSE né un migliore punteggio fisico. Un modello probabilistico sarebbe utile per scegliere una decisione che minimizzi la perdita attesa delle nostre metriche; richiede una distribuzione calibrata e inferenza più costosa. Prima verificare se un piccolo ensemble di E2 o un modello a miscela spieghino già l’incertezza rilevante.

## 8. E4 — Una base che segue la propagazione invece degli assi della matrice

Il metodo attuale interpola nel tempo e nello spazio e usa vicini a indici fissati. La fila dei sensori non ha distanze uniformi e i fronti possono muoversi tra un istante e il successivo.

L’**Adaptive Smoothing Method** combina interpolazioni spazio-temporali lungo direzioni compatibili con traffico libero e congestionato. È stato concepito proprio per ricostruire traffico da rilevatori stazionari sparsi. [Treiber e Helbing, 2002, archivio istituzionale](https://www.research-collection.ethz.ch/entities/publication/ec87f2bc-8df1-49ef-a3fc-0f3119f6c84c), [descrizione degli autori](https://www.akesting.de/trafficstates/?lang=en&site=methode).

**Adattamento proposto:** stimare sul train le direzioni di correlazione dei residui, dopo aver tolto livello del tratto e profilo periodico. Costruire poche interpolazioni lungo queste direzioni e usarle come basi o caratteristiche aggiuntive di LightGBM. Un kernel locale può pesare separatamente distanza e disallineamento temporale rispetto alla direzione di propagazione.

Non assumere che la velocità d’onda teorica del diagramma fondamentale descriva i dati sintetici rilasciati. Prima verificare che le correlazioni ritardate supportino una direzione utile. Se la correlazione tra vicini è alta soltanto perché condividono il profilo giornaliero, un kernel di propagazione può non aggiungere nulla.

**Limite:** quando tutti i sensori sono oscurati per 90 minuti, l’ASM non crea informazioni dentro il buco. Può sfruttare meglio gli ingressi e le uscite e aiutare nei buchi parziali. Misurare separatamente questi casi. Verificare l’unità di `milepost` prima di trasformare distanze in tempi di viaggio.

## 9. E5 — Un modello globale piccolo per le fluttuazioni di flusso

Il README riporta fluttuazioni reali del flusso fortemente correlate tra tratti vicini. Questo suggerisce una rappresentazione a pochi fattori, complementare alla correzione locale.

**Prima prova:** modellare il flusso per corsia come profilo di calendario più pochi fattori dinamici. Stimare il livello e i fattori del mese dalle osservazioni disponibili, con un prior temporale e shrinkage verso il train. Usare la previsione come caratteristica di E2 o LightGBM, evitando inizialmente di sostituire direttamente tutto il ramo.

La **Bayesian Temporal Factorization** combina fattorizzazione a basso rango e dinamica autoregressiva per imputazione e previsione di serie multidimensionali. Il lavoro tratta anche dati spazio-temporali. L’idea utile è il doppio vincolo: condivisione globale e dinamica locale. [Chen e Sun, versione 2021](https://arxiv.org/abs/1910.06366).

Per una prima verifica non serve necessariamente un campionatore bayesiano completo: un piccolo modello deterministico regolarizzato può dire se il segnale dei fattori è complementare. Non confrontarlo con modelli cui si fornisce meno contesto rilasciato.

**Problema da evitare:** un’intera colonna temporale senza alcun sensore osservato non è identificabile da una fattorizzazione priva di dinamica. Aumentare il rango non lo risolve. Servono prior temporali, osservazioni ai bordi e ripetizioni tra giorni; gli incidenti rari rimangono difficili.

**Criterio di successo:** riduzione degli errori comuni sul flusso, soprattutto nei contesti dove il vicino interpolato fallisce, senza attenuare eccessivamente le code. Tagliare dimensioni e finestre di elaborazione per contenere memoria e tempo sulla CPU condivisa.

## 10. E6 — Task 2: scegliere l’insieme che massimizza l’IoU attesa

Le soglie 0,40 e 0,45 hanno funzionato bene nella CV. Non è però detto che una soglia unica sia la decisione migliore per finestre con una sola cella in coda e finestre con un intero fronte esteso.

**RankSEG** formalizza la differenza tra classificazione puntuale e decisioni per Dice/IoU e propone regole basate su ranking e scelta della dimensione dell’insieme. Il benchmark è segmentazione, non traffico; le garanzie teoriche non si trasferiscono automaticamente alle forti dipendenze tra le nostre celle. [Dai e Li, JMLR 2023](https://www.jmlr.org/papers/v24/22-0712.html).

### Prova a basso costo sulle probabilità OOF

1. Conservare tutte le probabilità, con identificatori di finestra, pannello, orizzonte ed eleggibilità.
2. Calibrarle usando soltanto i blocchi interni, separando onset e ongoing se necessario.
3. Ordinare le celle di ogni finestra e confrontare diversi numeri di celle previste, invece di un unico cutoff globale.
4. Stimare la qualità attesa dei candidati con una regola dichiarata in anticipo, confrontandola con la soglia fissa.

Il rapporto tra intersezione attesa e unione attesa non è in generale l’IoU attesa: una scorciatoia di questo tipo deve essere etichettata come approssimazione. In alternativa, usare poche traiettorie congiunte calibrate dal train, come in E7, per valutare i candidati.

Usare il vincolo noto “almeno una coda nell’orizzonte”, che il metodo attuale già sfrutta. Per onset conservare la proprietà documentata T+30; non aggiungere arbitrariamente previsioni sui primi cinque passi. Per ongoing scegliere un insieme nell’intero orizzonte, senza forzare la stessa cardinalità per passo.

**Valutazione:** finestre ongoing ufficiali del train, selezioni casuali con più semi, origine progressiva e bootstrap per giorno/episodio. La scelta tra decoder si fa sui dati train antecedenti alle finestre di gara, non sul loro punteggio pubblico. Non è stato calcolato qui un nuovo risultato.

## 11. E7 — Task 2: rappresentare durata, nascita e propagazione delle code

Il classificatore corrente include il passo di previsione, ma non impone che le sei risposte siano una traiettoria coerente. Un tratto può alternare coda/non coda in modo implausibile o un fronte può spostarsi troppo rapidamente senza una penalità.

**Esperimento minimo:** usare le probabilità LightGBM come evidenza e aggiungere una penalità morbida per cambi di stato temporali e discontinuità spaziali. Stimare durata e transizioni sul train, distinguendo comparsa, persistenza e dissoluzione. Un modello a stati o un decoder su pochi candidati può bastare, prima di una rete più grande.

La penalità deve dipendere dalla distanza reale e consentire più code separate. Un solo blocco contiguo o l’ipotesi “una coda, una volta nata, rimane per tutti i sei passi” sarebbe troppo rigida. Gli incidenti possono creare code brevi o colli di bottiglia diversi.

Un’ulteriore prova è prevedere direttamente posizione del fronte e probabilità di dissoluzione, poi derivare gli stati per tratto. È un’ipotesi di modello da confrontare con il decoder più semplice.

La procedura di selezione è informazione rilasciata: il thread locale 745700 ammette condizionare una previsione su spaziatura e massimo IoU della persistenza. Si può quindi studiare una distribuzione di traiettorie compatibili con i vincoli noti, senza conoscere le etichette nascoste. Il limite resta usare soltanto origini e dati disponibili entro T, non le finestre future dello stesso file.

Se si addestra una piccola rete spazio-temporale, confrontare cross-entropy per cella con una loss sensibile all’IoU **per finestra**, non soltanto sull’intero batch. La Lovász-Softmax offre un riferimento per questo tipo di obiettivo; nel caso binario occorre la variante appropriata. I risultati su immagini non sono una promessa di miglioramento sulle code. [Berman et al., CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html).

**Criterio di successo:** il decoder migliora l’IoU su finestre indipendenti senza perdere gli episodi corti. Un diagramma più regolare non basta.

## 12. Un selettore basato sulla maschera può superare una regione temporale fissa

v10 sceglie il modello dei buchi per tutti i bersagli da T a T+24 e lascia il resto invariato. È una regola semplice, trasferita e già sostenuta dai dati. Non la sostituirei senza un confronto diretto.

La nuova ipotesi è che la scelta migliore dipenda da geometria e informatività del contesto, non soltanto dalla distanza dall’origine. Una cella a T+20 può avere buone misure vicine, mentre una cella esterna alla regione può appartenere a un altro lungo guasto.

**Prova:** combinare modello standard, specialista e una base robusta usando poche caratteristiche della maschera: lunghezza del buco, distanza dai bordi osservati, quota di sensori contemporaneamente disponibili, condizione onset/ongoing quando rilasciata, incertezza delle componenti. Cominciare da una regola o una regressione piccola, addestrata su predizioni OOF degli esperti.

Non addestrare il selettore sugli errori in-sample dei modelli che deve scegliere. Non usare il vero regime di congestione o l’errore vero come caratteristiche di inferenza. Valutare l’effetto completo ai confini, non soltanto nelle celle che il selettore cambia.

Questo esperimento ha senso soltanto se esistono almeno due esperti con errori complementari. Una media di modelli che sbagliano allo stesso modo attenua la varianza ma non recupera il contesto mancante.

## 13. Validare la maschera, non soltanto il modello

La prossima batteria dovrebbe incrociare tre assi, mantenendo uguali gli scenari tra candidati:

| Asse | Condizioni da mantenere |
|---|---|
| Osservabilità | R1/R2/R3, storie protette, blackout di corridoio, guasti parziali, bordi dei buchi |
| Dinamica | Flusso libero, onset, ongoing, dissoluzione, incidenti e giornate ordinarie |
| Trasferimento | Mesi distinti, livello spostato, variazioni spazialmente correlate, osservazioni rilasciate rimascherate |

Usare le maschere ufficiali simulate come scenario principale. Lunghezze del buco diverse da 90 minuti servono come stress test della dipendenza dal pattern, non come descrizione dei dati ufficiali. Non scegliere la durata che favorisce il candidato.

La generazione di scarti indipendenti gaussiani per tratto resta un test utile ma incompleto. Il mondo reale della gara può cambiare anche correlazioni, durata degli incidenti, disponibilità dei sensori e posizione dei colli di bottiglia. Una robustezza dimostrata soltanto sullo stesso generatore usato nell’augmentazione è evidenza debole.

La rimascheratura sui mesi rilasciati può sostenere adattamento self-supervised del Task 1. Per confrontarlo onestamente bisogna riservare celle osservate, nasconderle prima di costruire tutte le caratteristiche e non usarne i valori nell’adattamento che verrà valutato. Non basta cancellarle soltanto nell’ultimo passaggio.

**Noise2Self** fornisce una giustificazione per certi obiettivi di denoising da osservazioni rumorose quando il rumore è condizionalmente indipendente e il predittore non usa il valore da prevedere. Queste ipotesi sono plausibili per una parte del rumore di velocità in flusso libero; non vanno estese automaticamente alle fluttuazioni reali e correlate del flusso. [Batson e Royer, ICML 2019](https://proceedings.mlr.press/v97/batson19a.html).

## 14. Budget del punteggio: ordinare gli esperimenti con i pesi reali

La parte controllata da v e q è:

`0,35·S_state + 0,05·S_FD + 0,10·S_LWR`.

| Miglioramento ipotetico, a parità del resto | Delta sul totale |
|---|---:|
| S_queue +0,01 | +0,0030 |
| S_state +0,01 | +0,0035 |
| S_LWR +0,01 | +0,0010 |
| S_FD +0,01 | +0,0005 |
| S_ODME +0,01 | +0,0020 |

Per S_state, se tutti i regimi e pannelli migliorassero nello stesso modo e i termini non fossero al limite, ridurre l’RMSE della velocità di 1 km/h varrebbe circa 0,00756 sul totale; ridurre l’RMSE del flusso per corsia di 10 veicoli/h varrebbe circa 0,00268. Non si può applicare direttamente questo calcolo a una media dell’RMSE sul solo buco: la combinazione degli errori quadratici richiede i conteggi e gli errori del resto del dominio.

I valori della tabella sono sensibilità aritmetiche, non previsioni. Non conosciamo separatamente tutti i punteggi privati; non ricaverei un “margine certo” sottraendo i numeri locali da 1.

L’esperimento v10 ha prodotto circa +0,0035 modificando il 2,6% dei bersagli. Non segue che estendere la stessa correzione al resto dia un guadagno proporzionale: il modello dei buchi applicato ovunque era già peggiore sui casi normali.

## 15. Task 4 e strade che non metterei al primo posto

**Task 4:** mantenere la NNLS di riferimento. I conteggi non identificano tutte le componenti OD e i termini nascosti non hanno una validazione locale completa. Le due prove pubbliche sfavorevoli non dimostrano il meccanismo del riferimento, ma sono sufficienti per non spendere ora il budget su una ricerca di λ in classifica.

Per un’idea nuova, richiedere una verifica su problemi sintetici indipendenti con domanda nota, rango analogo e protocolli dichiarati, prima di considerare un invio. Anche un successo su queste simulazioni sarebbe soltanto evidenza supplementare.

Non metterei al primo posto:

- Altri alberi o più seed come rimedio al cambio di contesto: possono ridurre varianza, ma non ricreano osservazioni mancanti o maschere non viste.
- PINN o conservazione rigida basata su rampe e topologia: i flussi di confine che rendono informativa la metrica ufficiale non sono rilasciati; imporre un’altra equazione può peggiorare le risposte.
- Un grande foundation model temporale senza confronto con un modello piccolo addestrato sulle maschere corrette: cambiano dati, meccanismi e costo, senza evidenza di vantaggio qui.
- Una diffusione completa come primo esperimento: la distribuzione delle traiettorie è interessante, ma prima bisogna dimostrare che la sua incertezza aggiunge valore alla decisione finale.
- Smoothing indiscriminato di v, q o delle code: può cancellare il segnale che il valutatore premia.
- Riutilizzare nel Task 2 un’imputazione offline del Task 1: introdurrebbe dipendenze da timestamp successivi a T.

## 16. Sequenza operativa e criteri di arresto

1. **Riferimento v10 su pseudo-mesi completi.** Stessa inferenza dell’invio, tutte le predizioni reali ai bordi, stessi scenari per ogni candidato. Conservare separatamente gli errori per evento e regime.
2. **Due prove contenute:** E1 equivalente al livello, distinta dalle caratteristiche mensili già in test; E6 ranking/IoU sulle probabilità OOF. Una piccola griglia per ciascuna, scelta internamente.
3. **Un solo nuovo modello di sequenza:** E2 su blackout, prima senza e poi con E3. Confrontarlo con v10 e interpolazione sulle stesse patch e sullo stesso pseudo-mese.
4. **Un ramo complementare:** E4 se le correlazioni ritardate lo giustificano; altrimenti E5 sui fattori di flusso. Non provare entrambi soltanto per aggiungere varietà.
5. **Task 2 strutturato:** E7 solo dopo aver separato il guadagno da calibrazione, cardinalità e regolarità temporale.
6. **Selezione degli esperti e consegna:** tenere soltanto rami che migliorano la procedura completa, poi congelare codice, configurazioni, dati e pesi e preparare il report.

Per ogni candidato registrare delta appaiato per episodio/giorno, distribuzione tra pannelli e regimi, minimi, tempo e memoria. Il bootstrap deve ricampionare giorni o episodi, non celle; nuove maschere dello stesso giorno non aumentano il numero di eventi indipendenti.

Non confermare una modifica solo perché supera v10 in una media regionale o in un singolo mese rimascherato. Se il vantaggio sparisce con le predizioni reali fuori regione, o dipende da un solo corridoio o da una sola selezione di finestre, ridurre la complessità e fermare quel ramo.

La verifica delle frequenze e quella della metrica non vanno ripetute a ogni candidato se non cambia il relativo percorso. Le nuove verifiche devono rispondere ai rischi introdotti dal candidato, lasciando risorse agli esperimenti informativi.

## 17. Bibliografia operativa, integrazioni e limiti

| Fonte primaria | Uso proposto | Limite da rispettare |
|---|---|---|
| [RevIN, ICLR 2022](https://openreview.net/pdf?id=cGDAkQo1C0p) · [codice](https://github.com/ts-kim/RevIN) | Normalizzazione reversibile del livello | Statistiche da osservazioni valide; causalità distinta per Task 2 |
| [Noise2Self, ICML 2019](https://proceedings.mlr.press/v97/batson19a.html) | Adattamento e denoising senza verità pulita | Indipendenza del rumore e valore target escluso dagli ingressi |
| [SAITS, ESWA 2023](https://arxiv.org/abs/2202.08516) · [codice](https://github.com/WenjieDu/SAITS) | Imputazione di patch multivariate con maschere | Addestrare anche su blackout dell’intero corridoio |
| [GRIN, ICLR 2022](https://arxiv.org/abs/2108.00298) · [codice](https://github.com/Graph-Machine-Learning-Group/grin) | Relazioni spazio-temporali tra sensori | Nessuna informazione contemporanea quando tutti i nodi sono oscurati |
| [Adaptive Smoothing, Treiber e Helbing, 2002](https://www.research-collection.ethz.ch/entities/publication/ec87f2bc-8df1-49ef-a3fc-0f3119f6c84c) · [metodo](https://www.akesting.de/trafficstates/?lang=en&site=methode) | Base lungo direzioni di propagazione | Verificare che la dinamica sintetica supporti quei ritardi |
| [Bayesian Temporal Factorization, versione 2021](https://arxiv.org/abs/1910.06366) | Fattori condivisi e dinamica temporale | Colonne completamente mancanti richiedono un prior temporale |
| [CSDI, NeurIPS 2021](https://arxiv.org/abs/2107.03502) | Distribuzione delle traiettorie, eventualmente in seconda fase | Costo del campionamento e decisione finale non equivalenti a plausibilità |
| [RankSEG, JMLR 2023](https://www.jmlr.org/papers/v24/22-0712.html) | Decoder orientato all’IoU per finestra | Dipendenze tra celle e calibrazione da verificare sul traffico |
| [Lovász-Softmax, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html) | Loss sensibile all’IoU per finestra | Non sostituisce una validazione indipendente del decoder |

Ho consultato anche il [repository ufficiale TrafficFlowBench](https://github.com/jacky850/trafficflowbench-public). Per le regole specifiche e le precisazioni del forum mi sono basato sulle copie locali già presenti; non ho effettuato una nuova verifica live delle pagine Kaggle né aggiornato il checkout.

Fonti locali principali: [README](README.md), [prima revisione](REVISIONE_AVVERSARIALE.md), [osservazioni precedenti](OSSERVAZIONI.md), [ricostruzione Task 1](src/tfb/task1.py), [CV buchi](scripts/cv_buchi.py), [CV mensile](scripts/cv_mese.py), [caratteristiche](src/tfb/caratteristiche.py), [rimascheratura](scripts/rimaschera_valutati.py), [metrica](src/tfb/metrica.py), [caratteristiche Task 2](src/tfb/task2_modello.py), [training Task 2](scripts/allena_task2_finale.py), [conformità](docs/conformita.md).

I manifesti correnti migliorano il congelamento dei modelli, ma il pacchetto finale deve includere anche uno snapshot del codice e il percorso per ricostruire caratteristiche e training. Registrare gli hash dei soli pesi non preserva il comportamento di funzioni che cambiano successivamente.

Non ho installato le integrazioni citate, scaricato pesi o misurato prestazioni. Le proposte sono adattamenti motivati dalle evidenze del progetto e dalla letteratura: nessuna pubblicazione citata dimostra già il guadagno sui blackout e sulle code di questa competizione.
