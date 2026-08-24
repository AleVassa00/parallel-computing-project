# Progetto - Sistemi di Calcolo Parallelo e Applicazioni

Il progetto verte sulla realizzazione di un nucleo di calcolo per il prodotto tra una matrice densa ed un multivettore.

## Autori

**Alessandro Vassallo (0366572)** 
**Andrea Galluzzi (0365026)** 


**Laurea Magistrale in Ingegneria Informatica**
*Sistemi Di Calcolo Parallelo E Applicazioni (A.A. 2025/2026)*
Università degli Studi di Roma Tor Vergata

## Build ed esecuzione

```bash
make                # bin/matmul_mpi e bin/test_index
make test           # test delle funzioni indice (non richiede MPI)
make check          # validazione MPI contro il seriale su piu' forme di griglia
make PREC=float     # ricompila in singola precisione
make KERNEL=<nome>  # seleziona src/kernel/<nome>.c come local_gemm
```

Esecuzione:

```bash
mpirun -np 4 ./bin/matmul_mpi -M 8000 -N 8000 -k 8 \
       --pr 2 --pc 2 --reps 10 --a-mode local
```

`--a-mode local` e' il comportamento predefinito: ogni processo genera il
proprio blocco di A dagli indici globali. Con `--a-mode global`, il grid rank 0
genera invece A completa e la distribuisce sulla griglia cartesiana 2D tramite
`MPI_Type_vector`. Le due modalita' producono gli stessi valori locali.

Sul server di dipartimento il binding e' obbligatorio, altrimenti le misure
sono rumore:

```bash
mpirun -np 20 --bind-to core --map-by core --report-bindings \
       ./bin/matmul_mpi -M 10000 -N 10000 -k 32 --pr 4 --pc 5 --reps 10 --csv
```

Opzioni principali: `-M -N -k`, `--pr --pc` (forma della griglia, default la
fattorizzazione piu' quadrata di P), `--reps --warmup`, `--seed`,
`--a-mode local|global`, `--check` (validazione contro il seriale), `--csv` /
`--csv-header` per la campagna di misura. `--help` per l'elenco completo.

Il broadcast di X viene eseguito una sola volta nel setup. Il valore `gflops`
usa esclusivamente la media dei tempi massimi fra rank di `local_gemm`:
`2*M*N*k / t_local_mean / 1e9`. `gflops_total` include invece anche la reduce
di Y ed e' riportato soltanto come metrica aggiuntiva.

## Struttura

| Cartella | Contenuto |
|---|---|
| `src/common` | tipo scalare (`double`/`float`), utility, timer |
| `src/index` | partizionamento a blocchi degli indici globali |
| `src/gen` | generatore riproducibile, funzione degli indici globali |
| `src/serial` | implementazione seriale di riferimento (oracolo) |
| `src/kernel` | `local_gemm`, un file per implementazione |
| `src/mpi` | griglia cartesiana, distribuzione, algoritmo `Y = AX` |
| `src/bench` | driver di misura e validazione |
| `test` | test isolato delle funzioni indice |
