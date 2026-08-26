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
make check          # check-cxx + validazione MPI contro il seriale
make check-cxx      # kernel.h e util.h restano utilizzabili da nvcc
make PREC=float check # stessa matrice di validazione in singola precisione
make check-padding  # --a-mode global con lda = n_loc + 8
make KERNEL=<nome>  # seleziona src/kernel/<nome>.c oppure .cu come local_gemm
make KERNEL=cuda_naive check   # backend CUDA (richiede nvcc: solo sul server)
```

Ogni configurazione ha un binario con nome proprio, cosi' una build non puo'
sovrascriverne un'altra e far misurare in float una campagna che si credeva in
double. Il default (double, `scheme_a`) resta `bin/matmul_mpi`; ogni scostamento
aggiunge un suffisso:

| comando | binario |
|---|---|
| `make` | `bin/matmul_mpi` |
| `make PREC=float` | `bin/matmul_mpi-float` |
| `make FORCE_GENERIC_K=1` | `bin/matmul_mpi-generic` |
| `make TEST_A_PADDING=8` | `bin/matmul_mpi-pad8` |
| `make KERNEL=cuda_naive` | `bin/matmul_mpi-cuda_naive` |

La riga finale di `make` ricorda sempre quale binario e' stato prodotto e con
quali variabili.

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

Una invocazione del prodotto MPI esegue, nell'ordine, `MPI_Bcast` della fetta
di X lungo il column communicator, `local_gemm`, e `MPI_Reduce(MPI_SUM)` lungo
il row communicator. Generazione/distribuzione di A resta preprocessing.

Il kernel locale ha un ciclo di vita in tre fasi, perche' A non cambia mai fra
un'invocazione e l'altra mentre X e Y cambiano sempre:

```c
kern = local_gemm_create(m_loc, n_loc, k, A_loc, lda);  /* preprocessing, non cronometrato */
local_gemm(kern, X_loc, ldx, Ypart, ldy);               /* regione cronometrata */
local_gemm_destroy(kern);
```

Per `scheme_a` create/destroy si limitano a registrare forma e puntatore. Per
il backend CUDA `create` e' la `cudaMemcpy` H2D di A in VRAM, che avviene una
volta sola: la consegna esclude dalla misura il tempo di trasferimento da e
verso la scheda, e con A dell'ordine dei GB una copia per invocazione
misurerebbe il PCIe, non la GPU.

## Backend CUDA

Il Makefile riconosce da solo se il backend e' C o CUDA: se esiste
`src/kernel/<nome>.cu` viene compilato con `nvcc -arch=sm_75` e il binario
linkato con `-lcudart`. Non c'e' nessun flag da ricordarsi.

```bash
make KERNEL=cuda_naive check      # 24 validazioni contro il seriale, su GPU
mpirun -np 1 ./bin/matmul_mpi-cuda_naive -M 10000 -N 10000 -k 32 --reps 10
```

Su una macchina senza `nvcc` la build si ferma con un messaggio esplicito: il
backend CUDA si compila sul server (`module load cuda`, oppure
`make NVCC=/usr/local/cuda/bin/nvcc ...`). Tutto il resto del progetto, incluso
`make check-cxx`, continua a funzionare ovunque.

`cuda_naive` e' la **baseline**, non il kernel finale: un thread per elemento di
Y, che quindi perde il riuso di A in registro dello schema A. Serve a validare
la pipeline e a dare il numero contro cui misurare le versioni successive.

Le fasi cadono cosi':

| fase | dove | cronometrata? |
|---|---|---|
| creazione contesto CUDA, `cudaMalloc`, H2D di **A** | `local_gemm_create` | no, e' preprocessing (`t_setup_s`) |
| H2D di **X**, kernel, D2H di **Y** | `local_gemm` | si', e' `t_local` |
| solo il kernel, misurato con i `cudaEvent` | dentro `local_gemm` | si', ed e' `t_kernel` |

`t_local - t_kernel` e' esattamente il costo del PCIe per invocazione, che la
consegna consente di riportare a parte anziche' dentro T.

I buffer di X e Y si allocano alla prima invocazione, perche' `ldx` e `ldy`
arrivano da li' e non da `local_gemm_create`. Con il default `--warmup 2` quella
prima invocazione e' un warm-up e l'allocazione resta fuori dalle repetition
cronometrate; il suo costo viene comunque sommato a `t_setup_s`, quindi non
sparisce dai dati. Con `--warmup 0` finirebbe nella prima repetition.

Se il blocco locale non entra in VRAM (40000x40000 in double sono 12.8 GiB
contro i 15.5 GiB della Quadro RTX 5000) il programma si ferma prima di
allocare, dicendo quanta memoria serve e quanta ce n'e'.

Per ogni repetition si prende separatamente il massimo fra i rank dei tempi di
broadcast, calcolo locale, reduce e totale. Il totale e' misurato direttamente
dall'inizio del broadcast alla fine della reduce. Le metriche sono:

```text
gflops         = 2*M*N*k / mean(max_rank(t_total))  / 1e9   metrica ufficiale
gflops_compute = 2*M*N*k / mean(max_rank(t_local))  / 1e9   senza le collettive
gflops_kernel  = 2*M*N*k / mean(max_rank(t_kernel)) / 1e9   senza H2D/D2H
```

`t_kernel` e `gflops_kernel` valgono `-1` per i backend di CPU, dove il kernel
coincide con `t_local` e non c'e' niente da separare. `t_setup_s` riporta il
preprocessing del backend, che per costruzione sta fuori da tutte e tre.

`scheme_a` specializza in C portabile `k=3,6,8,20,32` e usa il kernel generico
per ogni altro valore. Per un microbenchmark sullo stesso problema:

```bash
make FORCE_GENERIC_K=1        # i due binari coesistono, non serve ricompilare
mpirun -np 4 ./bin/matmul_mpi         -M 8000 -N 8000 -k 8 --pr 2 --pc 2 --csv
mpirun -np 4 ./bin/matmul_mpi-generic -M 8000 -N 8000 -k 8 --pr 2 --pc 2 --csv
```

## Struttura

| Cartella | Contenuto |
|---|---|
| `src/common` | tipo scalare (`double`/`float`), utility, timer |
| `src/index` | partizionamento a blocchi degli indici globali |
| `src/gen` | generatore riproducibile, funzione degli indici globali |
| `src/serial` | implementazione seriale di riferimento (oracolo) |
| `src/kernel` | `local_gemm`, un file per implementazione (`.c` o `.cu`) |
| `src/mpi` | griglia cartesiana, distribuzione, algoritmo `Y = AX` |
| `src/bench` | driver di misura e validazione |
| `test` | test isolato delle funzioni indice, probe C++ dell'interfaccia |
