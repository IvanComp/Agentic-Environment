# Progettazione di un contributo ICSA su architetture multi-agente

## Ruolo

Agisci come un ricercatore senior in Software Architecture, sistemi multi-agente basati su LLM ed empirical software engineering.

Il tuo obiettivo è aiutarmi a definire un contributo scientifico forte e difendibile per ICSA e a trasformare l'idea selezionata in una piattaforma sperimentale funzionante. Il lavoro deve produrre conoscenza utile per i Software Architect, non limitarsi a confrontare le prestazioni di alcuni LLM.

Il risultato del lavoro comprende due deliverable principali, entrambi obbligatori e tra loro coerenti:

1. un PDF che sviluppi la teoria, il framing architetturale, le direzioni considerate, l'approccio selezionato e il piano di evaluation;
2. il codice modificato di `AgenticQuantFramework`, pronto per eseguire gli esperimenti necessari a verificare empiricamente e valutare l'approccio descritto nel PDF.

Il PDF non può limitarsi a descrivere il software e il codice non può limitarsi a dimostrare una possibilità tecnica. Il primo deve formulare il contributo scientifico; il secondo deve rendere quel contributo osservabile, misurabile e valutabile empiricamente.

## Contesto

Un mio precedente lavoro sui sistemi multi-agente è stato rifiutato perché percepito soprattutto come un paper di AI, con un contributo insufficiente per la comunità di Software Architecture.

Il nuovo lavoro deve quindi portare a risultati utilizzabili nelle decisioni architetturali, come:

- trade-off e condizioni di applicabilità;
- effetti sui quality attributes;
- architectural tactics, pattern o anti-pattern;
- criteri per scegliere o configurare un meccanismo di coordinamento;
- strumenti di reasoning e tracciabilità per i Software Architect.

Come possibile punto di partenza empirico, posso confrontare tre meccanismi di coordinamento:

- voting;
- debate;
- role-based coordination.

Non assumere, però, che questi tre meccanismi debbano essere il centro del contributo. Prima ricostruisci il problema scientifico e architetturale; solo dopo stabilisci se e in quale ruolo i tre meccanismi siano pertinenti.

## Vincolo di originalità

Il nucleo dell'approccio proposto non deve essere già presente nella letteratura o in un framework esistente. È ammessa e necessaria una letteratura correlata, ma non è sufficiente riprodurre un metodo noto in un nuovo esperimento. La proposta deve offrire una novità sostanziale e dimostrabile per la Software Architecture.

Prima di consolidare l'idea, formulare claim definitive o modificare estesamente `AgenticQuantFramework`, applica il novelty gate descritto più avanti. Se emerge un approccio equivalente, non proseguire per inerzia: scarta o riformula la direzione e confrontati con me.

## Piattaforma software esistente

La piattaforma è già disponibile nella cartella:

`Scrivania/AgenticQuantFramework`

Accedi al repository e analizzalo prima di proporre modifiche. Se il percorso non è disponibile o non è univoco, chiedimi dove si trova il repository e non inventare la sua struttura.

`AgenticQuantFramework` è una base esistente ed estendibile che può istanziare LLM differenti ed eseguire benchmark tratti dalla letteratura. Il lavoro deve partire da questa implementazione: non progettare una piattaforma completamente nuova se non emerge una ragione tecnica o scientifica forte.

Prima di modificare il codice:

1. ricostruisci l'architettura attuale della piattaforma;
2. identifica componenti, punti di estensione, dipendenze e flusso di esecuzione degli esperimenti;
3. verifica quali meccanismi di coordinamento sono già implementati e con quali assunzioni;
4. controlla quali modelli, benchmark, metriche, configurazioni e modalità di logging sono supportati;
5. individua i limiti che impediscono di realizzare il disegno sperimentale proposto.

L'idea di ricerca e la piattaforma devono restare allineate. Dopo avere sviluppato e motivato la migliore direzione scientifica, modifica `AgenticQuantFramework` affinché diventi la piattaforma effettivamente utilizzabile per gli esperimenti del paper.

Le modifiche devono:

- implementare i costrutti e le variabili previsti dal disegno sperimentale;
- rendere configurabili i fattori sperimentali rilevanti;
- supportare baseline, repliche e ablation study;
- raccogliere dati sufficienti a rispondere alle research questions;
- garantire tracciabilità tra configurazione, esecuzione e risultato;
- favorire riproducibilità e analisi successive;
- preservare, per quanto possibile, le funzionalità già presenti;
- includere documentazione, configurazioni di esempio e test proporzionati alle modifiche.

Non aggiungere funzionalità solo perché tecnicamente interessanti. Ogni estensione deve derivare da una domanda di ricerca, da una variabile, da una misura o da un requisito di riproducibilità.

## Materiali da analizzare

I materiali principali si trovano in:

`/Users/ivan/Desktop/icsa27`

Analizza almeno i documenti seguenti, rispettando per ciascuno la funzione indicata.

### `icsa26-rejected.pdf`

Questo documento contiene le review del precedente contributo rifiutato e fornisce elementi per ricostruirne l'idea. Usalo per:

1. ricostruire l'idea di partenza, le research questions, il metodo e il contributo dichiarato;
2. identificare le ragioni esplicite e implicite del rifiuto;
3. distinguere i problemi di framing dai limiti teorici, metodologici, empirici e di validità;
4. capire perché il lavoro è stato percepito come un paper di AI e non di Software Architecture;
5. tradurre ogni critica rilevante in un requisito verificabile per il nuovo contributo;
6. individuare le parti dell'idea precedente che possono essere recuperate senza riprodurne i difetti.

Non assumere che ogni osservazione dei reviewer sia automaticamente corretta. Valutane la fondatezza usando il testo del paper, la letteratura e i criteri della comunità di Software Architecture. Prepara una tabella con: osservazione del reviewer, evidenza nel manoscritto, gravità, conseguenza per il nuovo lavoro e modifica necessaria.

### `Peer-Preservation in Frontier Models.pdf`

Questo è il riferimento principale per ricostruire il problema della peer preservation. Usalo per:

1. ricavare la definizione originale del costrutto;
2. identificare cause, meccanismi, condizioni e outcome osservati;
3. ricostruire il modello teorico ed eventuali procedure di misura;
4. distinguere ciò che il paper dimostra da ciò che non è stato ancora studiato;
5. valutare se il problema possa essere riformulato come problema o decisione di Software Architecture;
6. verificare se voting, debate e role-based coordination lo affrontino davvero.

Non estendere automaticamente i risultati del paper a sistemi, modelli o task differenti. Identifica con precisione i confini entro i quali le sue conclusioni sono valide.

### `icsa2026-accepted.pdf`

Questo paper ICSA 2026 accettato ha due funzioni distinte:

1. fornire un esempio concreto del tipo di problema, argomentazione, evidenza empirica e contributo architetturale che ha superato il processo di selezione di ICSA;
2. mostrare come gli Architecture Decision Records vengono impiegati in un contesto che combina AI e agenti.

Confrontalo con `icsa26-rejected.pdf` per identificare differenze osservabili in:

- centralità del problema architetturale;
- destinatari della conoscenza prodotta;
- ruolo degli ADR;
- rapporto tra artefatto tecnico e contributo scientifico;
- tracciabilità tra dati e implicazioni per gli architect;
- forza e delimitazione delle claim;
- disegno e qualità della valutazione empirica.

Non trattare questo singolo paper come rappresentativo dell'intera conferenza e non copiarne meccanicamente struttura o terminologia. Usalo come termine di confronto concreto. Distingui ciò che il testo dimostra da ciò che possiamo solo inferire sulle ragioni dell'accettazione.

### `Architecting AgentOps.pdf` e `Agentops2.pdf`

Usa questi due documenti per ricostruire lo stato dell'arte su AgentOps e stabilire:

1. come definiscono AgentOps e quale problema intendono risolvere;
2. quali componenti, lifecycle concern, quality attributes e decisioni architetturali introducono;
3. quali esigenze pongono in termini di osservabilità, valutazione, governance, tracciabilità e riproducibilità;
4. quali elementi possono orientare l'architettura e la strumentazione di `AgenticQuantFramework`;
5. se AgentOps offre un framing utile per rendere operative le decisioni sul coordinamento o sulla peer preservation;
6. quali gap restano aperti tra principi AgentOps, scelta del coordinamento e comportamento emergente dei sistemi multi-agente.

Confronta i due documenti e segnala convergenze, differenze e possibili incompatibilità. Non usare AgentOps come semplice etichetta architetturale. Se il collegamento con peer preservation, coordinamento o piattaforma sperimentale non è sostenuto, tienilo fuori dal contributo centrale e spiegane il motivo.

### Altre fonti e artefatti

Analizza inoltre:

- eventuali altri articoli, benchmark e risultati precedenti che ti fornirò;
- il codice, i test e la documentazione di `AgenticQuantFramework`;
- la letteratura aggiuntiva necessaria a verificare gap o affermazioni non coperti dai documenti principali.

### Regole per l'analisi delle fonti

Per ogni affermazione importante:

- indica il documento e, quando possibile, la pagina o la sezione;
- separa le evidenze presenti nelle fonti dalle tue interpretazioni e dalle nuove proposte;
- non inventare risultati, definizioni, riferimenti o dettagli mancanti;
- segnala ambiguità, contraddizioni e limiti;
- chiarisci quando una conclusione richiede ulteriore letteratura.

## Protocollo di interazione

Durante lo sviluppo dell'idea puoi e devi farmi domande quando un dubbio può cambiare in modo sostanziale:

- il problema di ricerca;
- la scelta tra direzioni alternative;
- le assunzioni sulla piattaforma;
- la fattibilità degli esperimenti;
- i modelli o benchmark disponibili;
- il ruolo degli ADR;
- il contributo rivendicabile nel paper.

Formula domande precise e spiega quale decisione dipende dalla risposta. Non interrompere il lavoro per dettagli che puoi verificare nei materiali, nel repository o tramite un'assunzione prudente. Rendi esplicite le assunzioni che adotti.

Se emergono più opzioni plausibili con conseguenze scientifiche differenti, presentami il confronto e chiedimi quale vincolo o priorità adottare prima di impegnarti in modifiche estese al codice.

## Fase 1: ricostruzione del problema

Parti dal paper sulla peer preservation, non dalla tassonomia voting/debate/role-based.

1. Definisci con precisione la peer preservation.
2. Chiarisci:
   - quale problema osservabile descrive;
   - quali cause vengono proposte;
   - quali proprietà del sistema compromette;
   - a quale livello si manifesta: agente, interazione, coordinamento, topologia o architettura complessiva;
   - in quali condizioni diventa rilevante.
3. Identifica stakeholder, quality attributes e decisioni architetturali coinvolte.
4. Determina se la peer preservation sia meglio intesa come:
   - quality attribute;
   - proprietà architetturale;
   - meccanismo;
   - rischio;
   - outcome emergente;
   - oppure un costrutto che richiede una classificazione diversa.
5. Distinguila da concetti vicini, tra cui diversity, autonomy, influence, dominance, consensus, robustness e fault tolerance.
6. Spiega quali parti del problema possono essere osservate o misurate con la piattaforma esistente e quali richiedono nuove funzionalità.

Solo dopo questa ricostruzione valuta se voting, debate e role-based coordination:

- affrontano direttamente il problema;
- lo influenzano indirettamente;
- possono esserne una causa;
- introducono trade-off rilevanti;
- sono soltanto trattamenti sperimentali;
- oppure non hanno un legame teorico abbastanza solido.

Se un collegamento non regge, scartalo e motivane l'esclusione.

## Fase 2: stato dell'arte e gap

Ricostruisci lo stato dell'arte necessario a sostenere il contributo:

- ciò che sappiamo sui meccanismi di coordinamento nei sistemi multi-agente basati su LLM;
- ciò che sappiamo sulla peer preservation;
- le evidenze disponibili sugli effetti architetturali del coordinamento;
- l'uso degli ADR in sistemi adattivi, agentici o basati su AI;
- il ruolo di AgentOps nell'architettura, nell'esercizio e nella valutazione dei sistemi multi-agente;
- la differenza tra risultati prestazionali e conoscenza architetturale;
- il confronto tra `icsa26-rejected.pdf` e `icsa2026-accepted.pdf`;
- il rapporto tra l'idea in esame e i quattro nuclei documentali: review precedenti, peer preservation, ADR in un paper ICSA accettato e AgentOps.

Organizza il gap distinguendo:

- gap teorico;
- gap empirico;
- gap metodologico;
- gap nel supporto alle decisioni architetturali;
- gap strumentale rispetto alle capacità attuali di `AgenticQuantFramework`.

Concludi indicando quali gap sono sostenuti dai materiali e quali richiedono una ricerca bibliografica ulteriore.

### Novelty gate: verifica che l'approccio non esista già

Il nuovo contributo non deve riproporre un approccio già presente nella letteratura cambiando soltanto modelli, benchmark, dataset, prompt o terminologia. Prima di selezionare una direzione e prima di modificare il codice, esegui una ricerca di anteriorità mirata per stabilire se esistano lavori sostanzialmente equivalenti.

La verifica deve coprire almeno:

- Software Architecture;
- sistemi multi-agente basati su LLM;
- meccanismi di coordinamento, consensus, debate, voting e role assignment;
- peer preservation e costrutti equivalenti o vicini;
- AgentOps;
- ADR applicati a sistemi AI o agentici;
- framework di supporto alle decisioni per la configurazione di agenti e coordinamento.

Non cercare soltanto le espressioni usate nei documenti iniziali. Costruisci query con sinonimi, concetti confinanti e combinazioni alternative. Consulta fonti accademiche adeguate, includendo almeno ACM Digital Library, IEEE Xplore, Scopus o Web of Science quando accessibili, Google Scholar e arXiv per i preprint recenti. Applica backward e forward snowballing ai lavori più vicini.

Documenta:

- data della ricerca;
- database consultati;
- query utilizzate;
- criteri di inclusione ed esclusione;
- lavori candidati e ragioni dell'esclusione;
- limiti della ricerca.

Per ogni lavoro vicino, prepara una matrice di confronto che includa:

- problema affrontato;
- unità di analisi;
- costrutto centrale;
- meccanismo proposto;
- livello architetturale interessato;
- destinatario del contributo;
- metodo di evaluation;
- artefatto prodotto;
- conoscenza trasferibile ai Software Architect;
- sovrapposizione con la nostra proposta;
- differenza residua e sua rilevanza scientifica.

Classifica ogni candidato come:

- non equivalente;
- parzialmente sovrapposto;
- equivalente nel nucleo;
- incerto per insufficienza di informazioni.

La novità non è sufficiente se consiste soltanto in:

- eseguire benchmark diversi;
- usare modelli più recenti;
- confrontare tre configurazioni non ancora confrontate insieme;
- applicare un metodo esistente a un nuovo dominio senza produrre nuova conoscenza architetturale;
- rinominare un costrutto già studiato;
- aggiungere ADR o AgentOps come elementi documentali senza una funzione nuova.

La proposta può superare il novelty gate solo se il nucleo del contributo introduce una differenza sostanziale e difendibile, per esempio una nuova teoria causale, un nuovo costrutto architetturale, un metodo originale di decisione, una tactic non nota, una relazione non studiata tra decisioni e outcome oppure un modo nuovo e valutabile di operazionalizzare il problema per i Software Architect.

Se trovi un approccio equivalente nel nucleo:

1. non procedere con la proposta invariata;
2. indicami il lavoro trovato e il grado di sovrapposizione;
3. valuta se esiste un gap residuo realmente sostanziale;
4. proponi una riformulazione oppure scarta la direzione;
5. fammi una domanda se la scelta tra riformulare e cambiare direzione dipende dalle mie priorità.

Non usare claim assolute come "primo lavoro" o "nessun lavoro esistente" se la ricerca non le sostiene. Formula la claim di novità in modo preciso, delimitato e verificabile.

## Fase 3: direzioni di contributo alternative

Proponi almeno tre direzioni di ricerca sostanzialmente diverse. Non presentare semplici variazioni dello stesso benchmark.

Per ogni direzione specifica:

- problema affrontato;
- tesi candidata;
- contributo scientifico;
- conoscenza destinata ai Software Architect;
- ruolo della peer preservation;
- ruolo di voting, debate e role-based coordination;
- ruolo degli ADR;
- uso ed estensioni richieste a `AgenticQuantFramework`;
- metodo di valutazione;
- novità rispetto allo stato dell'arte;
- esito del novelty gate e lavori più vicini;
- rischi teorici e metodologici;
- probabilità che venga percepita come un paper di AI;
- fattibilità con le risorse disponibili.

Confronta e classifica le direzioni in una tabella usando criteri espliciti:

- rilevanza per ICSA;
- novità architetturale;
- solidità teorica;
- valutabilità empirica;
- generalizzabilità;
- utilità per i practitioner;
- compatibilità con la piattaforma;
- costo di implementazione;
- rischio di rejection.

Assegna un punteggio motivato a ogni direzione e seleziona la migliore. Se nessuna è abbastanza forte, dichiaralo e proponi come riformulare il problema.

Escludi dalla classifica finale qualsiasi direzione il cui nucleo risulti già coperto da un approccio esistente, anche se sarebbe facile implementarla con la piattaforma disponibile.

Prima di implementare modifiche estese alla piattaforma, presentami la direzione selezionata, le alternative scartate, le ragioni della scelta e gli eventuali dubbi ancora aperti.

## Fase 4: sviluppo della direzione migliore

Per la proposta selezionata definisci:

1. problema di ricerca;
2. unità di analisi;
3. contesto e confini dello studio;
4. gap preciso;
5. tesi centrale;
6. contributi attesi;
7. research questions;
8. ipotesi falsificabili, quando appropriate;
9. costrutti teorici e definizioni operative;
10. variabili indipendenti, dipendenti, di controllo e contestuali;
11. possibili confondenti;
12. metriche e procedure di misurazione;
13. disegno sperimentale;
14. benchmark e criteri di selezione;
15. configurazioni di agenti e modelli;
16. baseline, repliche e ablation study;
17. strategia di analisi quantitativa e qualitativa;
18. minacce a construct, internal, external e conclusion validity;
19. condizioni nelle quali i risultati possono essere generalizzati.

Spiega come i risultati sperimentali verranno trasformati in conoscenza architetturale, per esempio attraverso:

- un decision framework;
- architectural tactics;
- pattern o anti-pattern;
- un catalogo di trade-off;
- un modello di qualità;
- linee guida sensibili al contesto;
- un metodo per selezionare o adattare il coordinamento.

Il contributo non può ridursi alla conclusione che un meccanismo ottiene un punteggio medio migliore degli altri.

## Fase 5: ruolo degli ADR

Valuta criticamente il ruolo degli Architecture Decision Records usando `icsa2026-accepted.pdf`, la letteratura specifica e, dove pertinente, i requisiti di tracciabilità emersi da `Architecting AgentOps.pdf` e `Agentops2.pdf`.

Stabilisci se gli ADR debbano essere usati come:

- oggetti empirici da analizzare;
- strumenti per documentare le configurazioni sperimentali;
- rappresentazioni strutturate delle decisioni;
- meccanismi di tracciabilità tra contesto, decisione e outcome;
- supporto operativo per i Software Architect;
- parte di un intervento sperimentale.

Non inserire gli ADR solo per avvicinare nominalmente il lavoro alla Software Architecture. Se il loro ruolo non è necessario o teoricamente giustificato, dichiaralo.

Se sono utili, proponi:

- uno schema ADR adatto al problema;
- le decisioni da registrare;
- alternative, quality attributes e trade-off da documentare;
- il collegamento tra ADR, configurazioni della piattaforma ed evidenze sperimentali;
- un modo per valutare se gli ADR migliorano davvero il processo decisionale.

## Fase 6: matrice dei risultati attesi

Costruisci una matrice che includa:

- risultato empirico possibile;
- interpretazione rispetto alla peer preservation;
- implicazione architetturale;
- decisione consigliata;
- condizioni di applicabilità;
- trade-off;
- livello di confidenza;
- conseguenze per research questions e ipotesi;
- dati che `AgenticQuantFramework` deve raccogliere per sostenere l'interpretazione.

La matrice deve includere risultati negativi o inattesi, come:

- nessuna differenza tra i meccanismi;
- effetti dipendenti dal modello;
- effetti dipendenti dal task;
- maggiore accuratezza ma minore peer preservation;
- maggiore peer preservation senza benefici prestazionali;
- interazioni tra coordinamento, topologia e composizione degli agenti.

## Fase 7: adeguamento di AgenticQuantFramework

Traduci il disegno sperimentale selezionato in un piano di modifica della piattaforma.

Il piano deve collegare in modo tracciabile:

`RQ -> ipotesi -> variabile -> misura -> evento o dato raccolto -> componente software -> analisi`

Per ogni modifica proposta indica:

- requisito scientifico che la giustifica;
- componente esistente interessato;
- cambiamento architetturale o implementativo;
- compatibilità con le funzionalità esistenti;
- test necessario;
- output prodotto;
- rischio per validità o riproducibilità.

Dopo avere chiarito i dubbi che possono alterare il disegno, implementa le modifiche nel repository esistente. Verifica almeno:

- esecuzione dei test;
- corretto funzionamento delle nuove configurazioni;
- produzione completa dei dati sperimentali;
- ripetibilità di una prova di esempio;
- assenza di regressioni evidenti;
- adeguatezza della documentazione per eseguire nuovi esperimenti.

La piattaforma finale deve essere realmente fruibile per gli esperimenti del paper, non un prototipo illustrativo scollegato dalle research questions.

## Fase 8: simulazione di una review ICSA

Assumi il ruolo di un reviewer ICSA scettico e prova a rigettare la proposta.

Considera almeno queste obiezioni:

- il lavoro è ancora un benchmark di LLM;
- il contributo architetturale è solo nominale;
- la peer preservation non è definita o misurata correttamente;
- i tre meccanismi non sono comparabili;
- gli ADR sono decorativi;
- i risultati non sono generalizzabili;
- manca una teoria che colleghi configurazione e outcome;
- le implicazioni per i practitioner non derivano dai dati;
- il contributo o un approccio sostanzialmente equivalente esiste già nella letteratura;
- la claim di novità dipende soltanto da nuovi LLM, benchmark o combinazioni sperimentali;
- la piattaforma incorpora assunzioni che influenzano artificialmente i risultati;
- l'artefatto software non consente una replica indipendente.

Per ogni obiezione:

1. assegna una gravità;
2. spiega perché potrebbe causare il rifiuto;
3. indica se la proposta attuale la supera;
4. suggerisci una modifica concreta;
5. stima il costo scientifico o implementativo della modifica.

Presenta quindi una versione rafforzata del contributo. Se un'obiezione non può essere risolta onestamente, dichiaralo.

## Risultati finali richiesti

Il lavoro è completo solo quando sono disponibili entrambi i deliverable descritti di seguito.

### Deliverable 1: rapporto scientifico in PDF

Produci un PDF autonomo e leggibile che teorizzi l'approccio e raccolga l'intero percorso di ricerca. Non deve essere una breve risposta, un insieme di appunti o una semplice esportazione della chat. Deve essere una relazione scientifica strutturata, utilizzabile come base per la stesura del paper ICSA.

Il PDF deve includere almeno:

1. executive summary;
2. contesto e motivazione;
3. analisi delle review contenute in `icsa26-rejected.pdf`;
4. confronto critico con `icsa2026-accepted.pdf`;
5. ricostruzione del problema della peer preservation a partire da `Peer-Preservation in Frontier Models.pdf`;
6. sintesi e confronto di `Architecting AgentOps.pdf` e `Agentops2.pdf`;
7. stato dell'arte e gap di ricerca;
8. protocollo ed esito della ricerca di anteriorità;
9. matrice dei lavori più vicini e dimostrazione della differenza sostanziale;
10. almeno tre direzioni di contributo sostanzialmente diverse, con valutazione comparativa;
11. motivazione della direzione selezionata e ragioni per cui le altre sono state scartate;
12. teoria o modello concettuale dell'approccio proposto;
13. problema, tesi, contributi, research questions e ipotesi;
14. costrutti, variabili, metriche, confondenti e disegno sperimentale;
15. ruolo degli ADR e relativo schema, se il loro impiego risulta giustificato;
16. matrice dei risultati possibili e delle rispettive implicazioni architetturali;
17. mapping tra concetti teorici, research questions, dati richiesti e funzionalità della piattaforma;
18. strategia di evaluation e analisi dei dati;
19. minacce alla validità e limiti;
20. simulazione della review ICSA e versione rafforzata del contributo;
21. decisioni prese, alternative scartate e assunzioni ancora aperte;
22. conclusione operativa per il paper.

La conclusione del PDF deve riportare in forma sintetica:

- titolo provvisorio;
- tesi in una frase;
- problema architetturale centrale;
- contributo architetturale principale;
- conoscenza o artefatto trasferibile ai Software Architect;
- strategia sperimentale;
- ruolo effettivo della peer preservation;
- ruolo effettivo di voting, debate e role-based coordination;
- ruolo effettivo degli ADR e di AgentOps;
- differenza rispetto a un paper di AI o a un semplice benchmark;
- motivo della rilevanza per ICSA;
- principale rischio residuo;
- punteggio di aderenza a ICSA da 1 a 10, con motivazione.

Nel PDF:

- cita fonti, pagine o sezioni per le affermazioni ricavate dai materiali;
- separa chiaramente evidenze, interpretazioni e proposte;
- usa tabelle e diagrammi quando migliorano la comprensione;
- inserisci una bibliografia coerente;
- usa testo selezionabile, numeri di pagina, indice e titoli leggibili;
- verifica visivamente il documento finale per evitare pagine tagliate, tabelle illeggibili, sovrapposizioni o caratteri mancanti.

### Deliverable 2: AgenticQuantFramework modificato e verificato

Modifica direttamente il codice del repository esistente `AgenticQuantFramework`. Questo codice sarà usato per verificare empiricamente e valutare l'approccio teorizzato nel PDF. Non fermarti alla proposta di un'architettura, a pseudocodice o a un piano di implementazione.

La piattaforma modificata deve:

1. implementare l'approccio selezionato e i meccanismi necessari a valutarlo;
2. rendere configurabili trattamenti, fattori di contesto, modelli, benchmark e repliche;
3. applicare baseline e ablation study definiti nel PDF;
4. raccogliere le variabili e le metriche necessarie a rispondere alle research questions;
5. registrare configurazione, seed, modello, prompt, interazioni, decisioni, tempi, errori e risultati quando pertinenti;
6. produrre output strutturati e analizzabili;
7. consentire la ripetizione degli esperimenti;
8. includere test automatici o controlli equivalenti per le nuove funzionalità;
9. includere configurazioni di esempio e istruzioni di esecuzione;
10. preservare le funzionalità esistenti oppure documentare e motivare ogni incompatibilità.

Esegui almeno una prova end-to-end di dimensione ragionevole che dimostri che:

- la configurazione viene caricata correttamente;
- il sistema esegue l'approccio previsto;
- i dati richiesti vengono raccolti;
- i risultati vengono salvati nel formato documentato;
- l'esperimento può essere ripetuto seguendo le istruzioni fornite.

Se l'esecuzione completa degli esperimenti del paper richiede costi, credenziali, modelli o tempi non disponibili, non simulare risultati. Completa e verifica comunque l'infrastruttura con test, mock o una prova ridotta chiaramente identificata. Indica con precisione cosa resta da eseguire e quali risorse richiede.

### Coerenza tra i due deliverable

Mantieni una matrice di tracciabilità con questa catena:

`contributo -> RQ -> ipotesi -> costrutto -> variabile -> metrica -> dato raccolto -> componente del tool -> analisi prevista`

Ogni costrutto empirico presentato nel PDF deve avere una misura e un punto di raccolta implementato nella piattaforma. Ogni nuova funzionalità del tool deve essere giustificata da una necessità scientifica documentata nel PDF.

Se durante l'implementazione emerge che una parte dell'approccio non è misurabile o tecnicamente realizzabile, non adattare silenziosamente il codice. Aggiorna il modello teorico e il disegno sperimentale, documenta il cambiamento e mantieni coerenti PDF e piattaforma.

### File accessori richiesti

Oltre ai due deliverable principali, produci:

- documentazione per installare e usare la piattaforma modificata;
- configurazioni di esempio;
- istruzioni per riprodurre la prova end-to-end e gli esperimenti completi;
- risultati dei test e della prova end-to-end;
- registro delle decisioni e delle alternative scartate;
- elenco trasparente delle attività non eseguite e delle ragioni.

## Definition of done

Non considerare il lavoro concluso se hai prodotto soltanto il PDF o soltanto il codice.

Il lavoro è concluso quando:

1. il PDF è stato generato, aperto, controllato visivamente e contiene l'approccio completo;
2. `AgenticQuantFramework` è stato effettivamente modificato nel repository esistente;
3. codice, configurazioni e documentazione sono coerenti con il disegno descritto nel PDF;
4. i test pertinenti sono stati eseguiti e i risultati sono riportati;
5. una prova end-to-end ha prodotto gli artefatti sperimentali attesi;
6. la matrice di tracciabilità non contiene costrutti o metriche privi di implementazione, salvo limiti dichiarati e motivati;
7. il novelty gate è stato completato e documentato, senza approcci equivalenti irrisolti;
8. la claim di novità è precisa, delimitata e sostenuta dal confronto con i lavori più vicini;
9. sono disponibili istruzioni sufficienti per proseguire con l'evaluation completa del paper.

## Criterio generale

Mantieni un atteggiamento critico. Non cercare di salvare a tutti i costi l'idea iniziale. Se i materiali non sostengono una connessione, se un costrutto non è misurabile, se il contributo rimane troppo vicino a un benchmark di AI o se il nucleo dell'approccio esiste già, evidenzialo e proponi un'alternativa più difendibile.

La qualità del risultato dipende dalla coerenza tra problema, teoria, research questions, esperimenti, dati raccolti, implicazioni architetturali e implementazione della piattaforma. Nessuno di questi elementi deve essere aggiunto solo per rendere il lavoro apparentemente più vicino a ICSA.
