# Relazione LaTeX

Sorgenti della relazione di progetto per Sistemi di Calcolo Parallelo e
Applicazioni, A.A. 2025/2026.

## Compilazione

```bash
./compila.sh            # -> main.pdf
./compila.sh watch      # ricompila a ogni salvataggio
./compila.sh clean      # rimuove gli ausiliari
```

oppure direttamente `latexmk -pdf main.tex`.

Pacchetti richiesti: tutti in una TeX Live / MacTeX standard
(babel-italiano, geometry, booktabs, siunitx, pgfplots, tcolorbox, listings,
hyperref, cleveref, microtype).

## Struttura

```
main.tex            documento principale, elenca i capitoli
compila.sh          wrapper su latexmk
preambolo.tex       pacchetti, stile, macro (\dato, \TODO, daverificare, ipotesi)
sezioni/            un file per capitolo
figure/             figure esterne (per ora vuota: i grafici sono in pgfplots)
```

| File | Contenuto |
|---|---|
| `00-frontespizio.tex` | copertina |
| `01-introduzione.tex` | problema, mappa requisiti→codice→relazione, ripartizione del lavoro |
| `02-problema.tex` | formalizzazione, schema A vs B, intensità aritmetica, generazione dati |
| `03-architettura.tex` | moduli, interfaccia `local_gemm`, ciclo di vita a tre fasi, build |
| `04-mpi.tex` | griglia 2D, indici, layout, distribuzione, modello di costo |
| `05-kernel-cpu.tex` | `scheme_a`, specializzazione su k, rilettura di X |
| `06-kernel-cuda.tex` | i quattro kernel CUDA + cuBLAS |
| `07-roofline.tex` | roofline GPU (FP64/FP32) e CPU |
| `08-metodologia.tex` | metriche, cronometri, campagne C1–C6, riproducibilità |
| `09-validazione.tex` | oracolo seriale, tolleranza ε√N, matrice di collaudo |
| `10-risultati.tex` | C3 con dati reali; C1, C2, C4, C5, C6 predisposte |
| `11-conclusioni.tex` | sintesi e lavori futuri |
| `A-interfaccia.tex` | opzioni CLI e colonne del CSV |
| `B-riproducibilita.tex` | comandi per riprodurre ogni risultato |

## Segnaposto

I punti che dipendono da campagne non ancora eseguite sono **visibili in rosso**
nel PDF, in tre forme:

- `\dato` — singolo numero mancante in tabella (rende `--` grigio);
- `\TODO{...}` — nota inline in rosso;
- `daverificare` — riquadro con l'elenco di cosa eseguire;
- `\figuradafare{...}` — riquadro delle dimensioni della figura definitiva, con
  la descrizione di che cosa deve mostrare.

L'ambiente `ipotesi` (riquadro azzurro) marca invece le interpretazioni proposte
ma non ancora confermate da un profilo, con l'indicazione di come falsificarle.

**Per la versione di consegna**: commentare la riga `\todovisibiletrue` in
`preambolo.tex`. I marcatori spariscono dal PDF e restano nel sorgente.

## Stato dei dati

L'unica campagna con dati reali è C3 (build specializzata, `scheme_a`, double,
10000×10000, P=1), presa da
`results/c3_cpu_baseline/c3_cpu_specialized.csv`. Tutte le altre sezioni di
`10-risultati.tex` sono già strutturate con tabelle e figure definitive: resta
da sostituire i segnaposto.
