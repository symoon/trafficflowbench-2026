# Revisione avversariale dell’approccio a TrafficFlowBench

Data: 19 ottobre 2026. Fotografia dei file locali durante lo sviluppo e i test dell’altro agente.

**Giudizio:** interpolazione con correzione LightGBM per il Task 1, classificatori separati per onset e ongoing nel Task 2 e NNLS per il Task 4 sono una base sensata. Il rischio maggiore è attribuire troppo valore alle stime locali: nel Task 2 una caratteristica supervisionata attraversa indirettamente i fold; nel Task 3 la metrica sostitutiva usa un universo di transizioni diverso da quello ufficiale. Prima di ampliare i modelli, consoliderei questi due punti e congelerei gli artefatti delle versioni inviate.

**Per questa revisione ho soltanto letto file e consultato fonti web. L’unico file scritto è questo documento. Non ho modificato codice, modelli, cache, dati o invii; non ho lanciato training, test, estrazioni o comandi Kaggle; non ho fermato processi né usato il server GPU.** Le azioni proposte sotto sono raccomandazioni per il lavoro successivo.

## Evidenze e priorità

| Elemento | Evidenza disponibile |
|---|---|
| Ultimo riferimento pubblico documentato | `v6`: 0,85232; punteggio riportato nel README, non ricontrollato tramite API |
| Task 1, fold 0 | Interpolazione 0,9054; LightGBM 800 alberi 0,9393 |
| Task 1, fold 1 | Interpolazione 0,9044; LightGBM 800 alberi 0,9389 |
| Task 1, fold 2 | Il log letto si ferma dopo il modello della velocità: verifica ancora parziale nello snapshot osservato |
| Task 2 | 4.196 finestre ricostruite; IoU locale dichiarata 0,865 onset e 0,914 ongoing |
| Task 2, natura delle etichette locali | Derivate dalle osservazioni train; quelle ufficiali derivano dallo stato sottostante |
| Task 3 | Flussi di confine ufficiali non disponibili; `s_lwr_stima` è una metrica sostitutiva |
| Task 4 | NNLS con prior dello split; KL e prior medio scartati dopo confronti pubblici |

| Priorità | Criticità | Azione consigliata |
|---|---|---|
| P0 | Frequenze di coda del training possono includere il fold esterno valutato | Ricostruire le frequenze dentro ogni fold esterno |
| P0 | Versioni nominali legate a modelli e cache modificabili | Manifesto e snapshot immutabile per ogni candidato |
| P1 | CV temporale bidirezionale per un compito di previsione futura | Affiancare validazione con origine progressiva |
| P1 | Finestre ricostruite diverse dalla selezione ufficiale | Verificare trasferimento per evento, condizione e procedura di selezione |
| P1 | Proxy LWR diverso dal valutatore e obiettivo incompleto | Allineare transizioni eleggibili e includere il termine FD |
| P1 | Training per cella, valutazione per regime/corridoio/finestra | Confrontare pesi coerenti con la metrica |
| P2 | Interpolazione e vicini non riflettono interamente distanze e propagazione | Caratteristiche fisiche mirate, valutate sugli incidenti |
| P2 | Inferenze troppo forti da statistiche marginali e due prove ODME | Esplicitare ciò che resta non identificabile |

## 1. Task 2: la frequenza storica attraversa indirettamente i fold

**Stato: dipendenza verificata nel codice; entità dell’ottimismo non misurata.**

In `scripts/cv_task2.py`, `costruisci_righe` prepara tre frequenze di coda. Per le finestre del blocco `k`, la frequenza è calcolata su tutti i giorni tranne quelli del blocco `k`. Solo dopo vengono costruiti i modelli esterni, allenati su due blocchi e valutati sul terzo.

Questo esclude correttamente le etichette del blocco di una finestra dalla sua frequenza. Non esclude però il blocco esterno da tutte le caratteristiche del training.

Esempio: il modello viene valutato sul blocco A e allenato su B e C. Le righe di B hanno una frequenza calcolata su A e C; quelle di C su A e B. Le etichette di A contribuiscono quindi alle caratteristiche con cui viene addestrato il modello che deve prevedere A.

È una contaminazione indiretta della validazione attraverso una caratteristica derivata dal target. Non dimostra che l’invio utilizzi dati successivi a T: nel codice di produzione le frequenze vengono dal train, che precede i mesi valutati. Le due questioni vanno distinte.

**Miglioramento:** per ogni fold esterno escludere il blocco valutato da ogni frequenza usata, comprese quelle delle righe di training. Dentro il training esterno si può poi usare una seconda suddivisione per costruire caratteristiche fuori campione. In una validazione con origine progressiva, una soluzione ancora più naturale è usare solo giorni precedenti all’origine.

Per isolare quanto conta questa dipendenza, confrontare almeno tre condizioni: frequenza attuale, frequenza costruita senza il fold esterno, nessuna caratteristica di frequenza. Tutto il resto deve restare identico. Un’ablazione sulla sola importanza LightGBM non misura questo rischio.

## 2. Task 2: separare blocchi temporali non equivale a prevedere il futuro

**Stato: differenza verificata nel protocollo; impatto da misurare.**

I tre fold allenano su due blocchi e valutano sul terzo. Quando si valuta giugno–agosto, il training include settembre–febbraio; quando si valuta settembre–novembre, include dicembre–febbraio.

Questo misura la generalizzazione a un periodo escluso con informazioni provenienti anche da mesi successivi. La consegna deve invece trasferirsi da giugno–febbraio a marzo e aprile. La CV attuale può essere utile come controllo di stabilità, ma non riproduce da sola questa direzione temporale.

**Miglioramento:** affiancare almeno due origini progressive, per esempio training su giugno–agosto e valutazione su settembre–novembre, poi training su giugno–novembre e valutazione su dicembre–febbraio. Frequenze, calibrazione e scelta delle soglie devono essere ricostruite nel training di ogni origine.

Per il Task 1 il contesto futuro della stessa giornata è consentito, perché è una ricostruzione offline. Questo non autorizza a interpretare una CV con mesi futuri nel training come misura esatta del trasferimento ai mesi nuovi.

Le soglie 0,40 e 0,45 hanno un plateau locale favorevole. Tuttavia sono state selezionate osservando gli stessi tre fold di cui si riporta la prestazione. Massimizzare il fold peggiore è una regola robusta, ma resta una selezione: serve un periodo esterno lasciato intatto per misurarne l’esito.

## 3. Task 2: 4.196 finestre non sono 4.196 eventi indipendenti del test

**Stato: differenze documentate nella selezione; distribuzione effettiva degli errori da misurare.**

La ricostruzione su tutti i nove mesi è più informativa delle sole finestre ufficiali di giugno. Gli organizzatori la descrivono come una scelta ragionevole nella copia locale del thread 748784. Ma non è una replica esatta della selezione nascosta.

Lo stesso modulo `finestre.py` documenta che la regola ricostruita ritrova 53 delle 80 origini ufficiali del train usando una spaziatura unica, contro 47 con la distanza per condizione. Non è una discrepanza trascurabile quando si sfrutta una proprietà così specifica della selezione come l’onset solo a T+30.

Inoltre:

- La selezione locale usa etichette da velocità osservate; quelle ufficiali derivano dallo stato sottostante. Vicino a `0,60 × v_f` il rumore può cambiare sia le etichette sia l’inclusione di una finestra.
- Lo storico di 60 minuti usa lo strato vero, correttamente più simile a `window_history` dello strato mascherato. La distribuzione della riga a T e dei suoi mancanti va però confrontata separatamente.
- Il test ha cinque finestre per condizione e pannello; il train ricostruito ne ha quantità molto diverse per pannello: da 138 per `D7_I10_W` a 728 per `D7_I405_N`.
- La spaziatura di sei ore limita le sovrapposizioni, ma non rende indipendenti giornate, colli di bottiglia e fenomeni ricorrenti.

**Miglioramento:** riportare IoU per finestra con data, origine, pannello, condizione, dimensione della coda e copertura; conservare questi identificatori anche nell’artefatto di CV. `cv_finestre.parquet`, come costruito dallo script corrente, salva pannello, condizione e blocco ma non l’identificatore e la data della singola finestra: questo limita bootstrap e diagnosi successive.

Usare ricampionamenti per giorno o episodio e simulare insiemi di cinque finestre per condizione, mantenendo l’aggregazione ufficiale. Confrontare anche selezione delle prime origini e altre finestre ammissibili. La proprietà onset a T+30 è sostenuta dalla risposta locale dell’organizzatore: non suggerisco di rimuoverla senza evidenza. Va però controllato che la sua applicazione alle finestre aggiuntive non introduca etichette incoerenti.

## 4. Il guadagno pubblico del Task 2 è inferiore a quello suggerito dalla CV

**Stato: confronto aritmetico tra numeri documentati, non nuova valutazione.**

Secondo il README, da `v1` a `v3` cambia soltanto il Task 2:

`0,83843 − 0,72215 = 0,11628`.

Se gli altri output sono effettivamente identici, questo corrisponde a un aumento pubblico di `S_queue` pari a `0,11628 / 0,30 = 0,3876`.

Dalle medie locali riportate:

- Persistenza: `(0,002 + 0,815) / 2 = 0,4085`.
- LightGBM: `(0,865 + 0,914) / 2 = 0,8895`.
- Guadagno locale: circa `0,481`, equivalente a `0,1443` sul totale.

Lo scarto rispetto al guadagno pubblico è quindi circa 0,0934 di `S_queue`, ossia 0,0280 sul totale. I numeri locali sono arrotondati; il confronto dipende dall’identità degli altri task. Non permette di isolare la causa e non ricostruisce il valore assoluto pubblico del Task 2.

**Interpretazione:** il trasferimento porta un vantaggio consistente, ma i numeri locali potrebbero essere ottimisti o riferirsi a finestre più favorevoli. Contaminazione delle frequenze, selezione delle finestre, rumore delle etichette e piccolo campione pubblico sono spiegazioni da distinguere.

Userei questo scarto come motivo per migliorare la validazione. Per il Task 2 eviterei di usarlo per regolare soglie o parametri sulla classifica, coerentemente con le indicazioni locali degli organizzatori sulla selezione del metodo con soli dati antecedenti a T.

## 5. Training per cella e punteggio per gruppo non hanno gli stessi pesi

**Stato: disallineamento verificato negli obiettivi; beneficio delle correzioni da valutare.**

### Task 1

LightGBM minimizza l’errore quadratico delle righe campionate. La metrica media invece i tre regimi con lo stesso peso e poi i pannelli nelle famiglie. Un regime con più celle mascherate, più giorni o più tratti contribuisce di più al training, ma non necessariamente al punteggio.

La preparazione aggiunge giorni completi e un campione di 80.000 celle per blocco e pannello. I giorni completi reintroducono quantità diverse di righe per regime e dimensione del pannello. La stratificazione è utile, ma non rende automaticamente uniforme il peso finale.

### Task 2

La log-loss binaria pesa ogni cella eleggibile. La metrica pesa ogni finestra, poi condizione, pannello e famiglia. Una finestra con molti link pesa di più nel training, mentre una coda piccola mal localizzata può perdere quasi tutta la sua IoU.

I due modelli separati evitano che i sei passi ongoing dominino direttamente il modello onset. Rimane lo sbilanciamento tra pannelli e tra dimensioni degli eventi all’interno di ogni condizione.

**Miglioramento:** confrontare pesi per pannello/regime nel Task 1 e per finestra/pannello nel Task 2, insieme a campionamento degli eventi difficili. L’inverso della numerosità è un primo candidato, non una replica esatta della media di RMSE o dell’IoU: la scelta va giudicata sulla metrica ufficiale aggregata.

Per il Task 2, confrontare anche una regola di selezione delle celle calibrata per l’IoU della finestra, oltre alla soglia fissa. Le probabilità marginali non descrivono tutte le dipendenze spazio-temporali: qualunque regola di questo tipo richiede una verifica esterna.

## 6. Task 3: il proxy locale non applica tutti i filtri del valutatore

**Stato: differenze verificate tra `tfb.metrica` e il valutatore ufficiale locale.**

`s_lwr_stima` sostituisce i flussi di confine mancanti con le variazioni vere di `N = (q/v)·L`. È una costruzione utile per misurare l’errore sulle variazioni della densità. Non è però equivalente al punteggio ufficiale.

Nel codice ufficiale, prima del calcolo LWR:

- vengono esclusi link con vicini interni necessari non osservabili;
- in modalità A/B vengono richieste le condizioni di validità delle rampe attaccate alla transizione;
- viene usato il residuo rispetto ai flussi di confine degli organizzatori, quando forniti;
- sono applicate le regole di ricostruzione dello stato, eleggibilità e durata delle transizioni.

Il proxy locale seleziona invece coppie temporali consecutive eleggibili su tutti i link del tensore, senza replicare i filtri di osservabilità topologica e validità delle rampe. Cambiano quindi numeratore, denominatore e peso relativo degli errori. Una correzione efficace sui link esclusi ufficialmente può sembrare vantaggiosa senza incidere altrettanto sul punteggio reale.

**Miglioramento:** riprodurre almeno l’universo delle transizioni ufficiali usando topologia e qualità delle rampe rilasciate. Conservare la sostituzione dei flussi ignoti come approssimazione esplicita e riportare due risultati: proxy attuale e proxy con filtri ufficiali. Misurare quante transizioni vengono escluse e dove si concentra il guadagno di ogni candidato.

Il punteggio perfetto del proxy vale 1 per costruzione. I documenti locali riportano un risultato fisico ufficiale perfetto inferiore a 1: non è possibile assumere che il delta locale si trasferisca uno a uno.

## 7. L’obiettivo Task 1/3 usato per la densità omette il termine FD

**Stato: omissione verificata nello script; importanza da quantificare.**

In `cv_densita.py`, la sintesi usa:

`0,35 × S_state + 0,10 × S_LWR`.

La parte completa del punteggio modificata da velocità e flussi è invece:

`0,35 × S_state + 0,05 × S_FD + 0,10 × S_LWR`.

Il README argomenta che il termine FD offre poco margine. Può essere vero per i candidati osservati, ma il suo valore non resta matematicamente costante quando si modificano flussi e velocità: la densità cambia, può cambiare il ramo del diagramma e cambia la normalizzazione del residuo.

**Miglioramento:** calcolare anche `S_FD`, che non richiede i flussi di confine nascosti, sul dominio coerente con il valutatore. La conclusione «densità diretta scartata perché vale solo +0,0003 sul totale» resta provvisoria finché il confronto omette un termine variabile e usa un proxy LWR non allineato.

Non raccomando comunque di adottare quella variante sulla base del piccolo guadagno dichiarato. Nello snapshot letto il log della densità è ancora parziale e non contiene la tabella finale dei 30 confronti citata nel README. È un limite dell’evidenza disponibile durante una lavorazione in corso, non una prova che il risultato descritto sia sbagliato.

### Dove cercare un miglioramento più mirato

Le code sono rare, ma la sensibilità di `N` alla velocità cresce molto quando la velocità è bassa:

`∂N/∂v = −qL/v²`.

Un errore piccolo su v in coda può quindi pesare molto sul termine LWR pur incidendo poco sull’RMSE globale. Due modelli separati per v e q non controllano direttamente questa amplificazione.

Proverei una correzione congiunta e regolarizzata di v e q, oppure un blend della densità condizionato al regime di congestione e all’incertezza del contesto. La decisione deve usare esclusivamente grandezze disponibili in inferenza, come velocità predetta e copertura: separare gli errori usando la velocità vera è utile per l’analisi, non è una regola utilizzabile nel test.

## 8. Task 1: la CV e il percorso finale differiscono nei casi di ripiego

**Stato: differenze verificate; frequenza dei casi non misurata.**

Quando mancano sia la base temporale sia quella spaziale:

- `cv_task1.py` sostituisce la base con la mediana delle basi disponibili per pannello e link, calcolata prima della divisione esterna;
- `allena_task1_finale.py` esclude dal training i residui la cui base non è finita;
- `ricostruisci_lgb` usa come ripiego la ricostruzione v1, che può arrivare al profilo storico per istante del giorno, e applica comunque la correzione LightGBM.

Non sono le stesse condizioni. La mediana della CV è calcolata su osservazioni disponibili, non sui target nascosti; non va chiamata automaticamente fuga di etichette. Include però il blocco valutato e non replica il ripiego finale né la sua distribuzione in training.

Inoltre, la CV principale non applica il clipping finale `v ≥ 3` e `q ≥ 50` usato dall’invio. `cv_densita.py` usa per il flusso per corsia il limite `50/4`, mentre la produzione applica 50 al flusso totale, quindi il limite per corsia dipende dal numero effettivo di corsie.

**Miglioramento:** usare la stessa funzione di base, ripiego e clipping in training, CV e invio. Riportare numero e perdita dei casi di ripiego per regime, lunghezza del buco e congestione. Per i casi non rappresentati nel training, confrontare una correzione attenuata o nulla con quella completa.

## 9. I vicini sono utili, ma l’indice non è una distanza fisica

**Stato: rappresentazione verificata; opportunità da valutare.**

Il metodo usa fortemente i rilevatori vicini. Nel Task 1 l’interpolazione spaziale avviene sull’indice dei tratti, non sulle loro posizioni; nel Task 2 i vicini da −4 a +4 sono definiti dal numero di rilevatori. Le caratteristiche aggiungono alcune distanze, ma un passo nell’indice può corrispondere a distanze fisiche molto diverse.

Ai bordi o attraverso un intervallo senza osservazioni, il vicino nell’array può non rappresentare una relazione dinamica utile. Anche i fronti di coda si muovono: guardare monte e valle allo stesso istante può attenuare o spostare un evento.

**Miglioramenti da confrontare:** interpolazione sulle coordinate longitudinali, indicatori dei grandi intervalli tra sensori, vicini in fasce di distanza e caratteristiche lungo direzioni spazio-temporali compatibili con propagazione e velocità d’onda. Prima di interpretare `pos_km` come chilometri, verificare esplicitamente l’unità del campo `milepost` da cui viene derivato.

Per il Task 2 ogni caratteristica deve restare causale e usare soltanto timestamp ≤ T. Per il Task 1 si possono usare anche osservazioni successive della stessa giornata. Una ricostruzione offline del Task 1 non deve diventare accidentalmente l’input online del Task 2.

## 10. Statistiche medie simili non escludono incidenti diversi

**Stato: limite del controllo di distribuzione esistente.**

`confronta_split.py` confronta velocità media, flusso, quantili e quota di coda sui dati osservati. È un controllo utile, ma incidenti rari con posizione e durata diverse possono lasciare quasi invariate queste statistiche e produrre gran parte degli errori.

Le celle osservate, inoltre, non sono un campione arbitrario delle celle nascoste: intervengono disponibilità dei sensori, maschere R1/R2/R3 e protezione delle finestre Task 2. Il confronto delle medie non dimostra l’equivalenza della distribuzione dei target.

**Miglioramento:** valutare sul train gli errori per eventi di congestione, fronti, durata dei buchi, R3, bordi della rete e fasce orarie. Affiancare almeno una validazione con episodi interi esclusi e un ricampionamento per giorno. In ogni regime bastano pochi giorni difficili per spostare la media: milioni di celle non significano milioni di esempi indipendenti.

Per il Task 2, eventuali descrizioni aggregate di validation/private devono rimanere diagnostica separata: non devono guidare input, adattamenti o selezione con informazioni posteriori alla T della finestra.

## 11. Task 4: non identificabilità e interpretazioni troppo forti

**Stato: limite strutturale riconosciuto nel progetto; alcune conclusioni superano l’evidenza.**

Quando l’incidenza sui link contati ha rango basso rispetto al numero di percorsi, molte domande OD riproducono gli stessi conteggi. Un `S_link` vicino a 1 non identifica il buon risultato su `S_od`, deviazione dal prior e attrazioni. Lo segnala anche il valutatore ufficiale locale.

Tenere la NNLS come riferimento dopo il peggioramento di KL e prior medio è una scelta ragionevole. Le due prove non dimostrano però che il riferimento nascosto sia generato da una soluzione NNLS con zeri. Dimostrano che quelle due alternative hanno reso peggio sul mese pubblico, assumendo identici gli altri task.

Anche l’argomento sulla correlazione dei log-prior richiede cautela. Se X, Y e Z sono rumori indipendenti con uguale varianza, allora:

`corr(Y − X, Z − X) = 1/2`.

La correlazione 0,50 può quindi nascere dal termine sottratto condiviso. Da sola non prova una particolare distribuzione del rumore, una domanda identica tra i mesi o il meccanismo del riferimento.

**Miglioramento:** mantenere la NNLS senza inseguire una griglia di λ sulla classifica. Documentare residuo sui conteggi, distanza dal prior, percentuale di zeri, attrazioni e sensibilità al condizionamento della matrice. Per nuove idee usare problemi sintetici indipendenti con OD nota, rumore plausibile e rango analogo, come verifica del metodo: non come sostituto della metrica nascosta.

L’uso della media dei prior di train, validation e private è confinato nel codice a una variante offline del Task 4. Non ho trovato un passaggio di questa informazione al Task 2.

## 12. Una versione come `v6` non è ancora uno snapshot immutabile

**Stato: rischio verificato nell’organizzazione degli artefatti.**

`genera_invio.py` associa le versioni a funzioni, ma queste caricano modelli da percorsi condivisi come `outputs/task1/finale_v.txt` e `outputs/task2/finale_queue_ongoing.txt`. Un nuovo training può sovrascriverli. Rigenerare `v6` in futuro può quindi produrre un risultato diverso dal file inviato come v6.

Con `--riusa`, invece, possono essere riutilizzate predizioni pregresse senza verificarne la corrispondenza con codice, dati e modelli correnti. Anche cache dei tensori e della chiave d’invio vengono accettate per esistenza del file, senza un collegamento all’hash del rilascio e all’ordine dei link.

Lo script presentato come ingresso unico genera le predizioni ma presuppone modelli già allenati e caratteristiche già disponibili. Non ricostruisce da solo tutte le fasi a partire dal rilascio pulito. Questo requisito è già riconosciuto in `docs/conformita.md`, ma non è ancora realizzato dal solo ingresso corrente.

**Miglioramento:** per ogni candidato conservare configurazione, hash del codice, dipendenze, rilascio, cache, ordine delle caratteristiche, modelli, CSV e identificativo Kaggle. Il riuso deve fallire se questi elementi non corrispondono. Preparare un ingresso documentato che ricostruisca anche caratteristiche e training.

La verifica di `verifica_task2_online.py` sulle 48 finestre è un buon controllo dell’inferenza. Non dimostra però la causalità dell’intero processo di selezione e training; inoltre confronta le decisioni binarie, che possono restare uguali anche se cambiano le probabilità. Il controllo finale dovrebbe partire da un processo pulito e confrontare caratteristiche, probabilità e decisioni dopo il taglio dei dati futuri.

Non suggerisco di eseguire ora questi controlli mentre l’altro agente è in test: sono verifiche finali da pianificare su una configurazione congelata.

## 13. La completezza dell’invio merita una verifica indipendente dalla cache

**Stato: copertura dei controlli verificata; nessun invio malformato riscontrato.**

Il writer verifica valori finiti, binarietà delle code, positività della velocità, non negatività dei flussi OD e unicità degli ID. I merge uno-a-uno sono una protezione utile.

Non confronta però esplicitamente l’insieme finale degli ID con il template originale: costruisce l’output dalla chiave eventualmente in cache. Una chiave obsoleta potrebbe quindi produrre un file internamente coerente ma incompleto rispetto al rilascio corrente. Le copie locali delle pagine Evaluation e Submission specificano che un task incompleto vale zero; alcuni documenti del repository descrivono ancora penalità per singola riga mancante.

**Miglioramento:** prima della consegna finale, verificare rispetto al template originale insieme degli ID, task, conteggi per task e split, valori nelle colonne pertinenti e assenza di extra. Controllare anche il file serializzato dopo l’arrotondamento a tre decimali. Non ho letto integralmente i CSV da milioni di righe né verificato un errore di questo tipo.

## Sequenza di miglioramento consigliata

1. **Congelare il riferimento attuale.** Conservare modelli e invio effettivo con un manifesto, prima che altri training li sovrascrivano.
2. **Rendere indipendente la CV del Task 2.** Frequenze dentro i fold esterni, soglie selezionate internamente e almeno un controllo con origine progressiva.
3. **Valutare l’evento, oltre alla cella.** Errori e incertezza per giorno, finestra, condizione, pannello e dimensione della coda.
4. **Allineare il Task 1 di CV all’inferenza.** Stessi ripieghi, clipping e composizione del training; completare i fold prima di consolidare le conclusioni.
5. **Raffinare il proxy fisico.** Filtri ufficiali, termine FD e diagnosi delle transizioni che producono il guadagno.
6. **Provare modifiche mirate.** Pesi coerenti con la metrica, correzione congiunta v/q nei casi congestionati e caratteristiche sulle distanze fisiche. Confronti appaiati su periodi ed eventi interi.
7. **Chiudere riproducibilità e report.** Rilancio pulito, manifesti, verifica degli ID e spiegazione dei limiti delle metriche locali. Questo produce anche evidenze utili per il report valutato dagli organizzatori.

La prima misura da ottenere è quanto dei circa 0,89 di IoU locale del Task 2 rimanga dopo aver escluso il fold valutato da tutte le frequenze. La seconda è quanto dei guadagni LWR sopravviva all’allineamento delle transizioni ufficiali. Sono controlli più informativi di un aumento immediato di alberi o di complessità.

## Fonti e limiti

Fonti locali principali:

- [README](README.md): strategia, risultati e confronti pubblici dichiarati.
- [CV Task 2](scripts/cv_task2.py), [finestre](src/tfb/finestre.py), [caratteristiche Task 2](src/tfb/task2_modello.py) e [inferenza Task 2](src/tfb/task2.py).
- [Preparazione Task 1](scripts/prepara_task1.py), [CV Task 1](scripts/cv_task1.py), [training finale](scripts/allena_task1_finale.py), [ricostruzione](src/tfb/task1.py) e [caratteristiche](src/tfb/caratteristiche.py).
- [Metrica locale](src/tfb/metrica.py), [esperimento densità](scripts/cv_densita.py) e [valutatore fisico ufficiale](ufficiale/src/task3/score_task3.py).
- [Task 4](src/tfb/task4.py) e [valutatore ODME ufficiale](ufficiale/src/task4/score_task4.py).
- [Generazione invio](scripts/genera_invio.py), [writer e controlli](src/tfb/invio.py), [cache dei dati](src/tfb/dati.py), [verifica online](scripts/verifica_task2_online.py) e [conformità](docs/conformita.md).
- [Log CV Task 1](outputs/task1/cv_blocchi.log), [log densità](outputs/task1/cv_densita.log) e [log CV Task 2](outputs/task2/cv.log). Gli artefatti sono fuori da git e possono cambiare durante le esecuzioni in corso.
- Copie locali dei thread [onset](docs/forum/742328.md), [validazione](docs/forum/748784.md), [storie protette](docs/forum/749278.md), [cutoff e selezione](docs/forum/748789.md), [topologia](docs/forum/743819.md), delle pagine [Evaluation](docs/kaggle/02_valutazione.md) e [Submission](docs/kaggle/03_invio.md).

Il [repository ufficiale consultato sul web](https://github.com/jacky850/trafficflowbench-public) conferma la separazione temporale train/validation/private e la generazione indipendente dei mesi valutati. Le pagine Kaggle tentate non hanno restituito contenuti leggibili: le precisazioni regolamentari citate provengono dalle copie locali e non sono una nuova verifica live del regolamento. Per l’analisi del codice ho usato il checkout ufficiale locale, senza aggiornarlo.

Nessun punteggio è stato ricalcolato in questa revisione. I numeri sono letti da log e documenti; le differenze di punteggio esplicitamente indicate sono semplici calcoli sui valori riportati. Le criticità di codice sono riscontri di lettura; i miglioramenti proposti non hanno ancora una misura sperimentale del beneficio.
