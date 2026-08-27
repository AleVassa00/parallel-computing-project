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
make KERNEL=cuda_warp check    # warp-per-row, template sui cinque k richiesti
make KERNEL=cuda_warp_smem check          # tile di X in shared memory
make KERNEL=cuda_warp_smem SMEM_PAD=0 check   # la stessa cosa senza padding
make KERNEL=cublas check       # riferimento esterno (aggiunge -lcublas da solo)
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
| `make KERNEL=cuda_warp` | `bin/matmul_mpi-cuda_warp` |
| `make KERNEL=cuda_warp_smem` | `bin/matmul_mpi-cuda_warp_smem` |
| `make KERNEL=cuda_warp_smem SMEM_PAD=0` | `bin/matmul_mpi-cuda_warp_smem-smempad0` |
| `make KERNEL=cublas` | `bin/matmul_mpi-cublas` |

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
kern = local_gemm_create(m_loc, n_loc, k, A_loc, lda, ldx, ldy); /* preprocessing */
local_gemm(kern, X_loc, ldx, Ypart, ldy);                        /* cronometrato */
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

`cuda_warp` assegna invece un warp a ogni riga di Y. La lane `l` visita gli
indici `j=l,l+32,...`, percio' le lane leggono consecutivamente A, e riusa ogni
valore caricato nei K accumulatori della lane. I risultati parziali vengono
ridotti nel warp con `__shfl_down_sync`; la lane 0 scrive la riga di Y. I casi
`k=3,6,8,20,32` sono istanze template distinte, mentre ogni altro k attraversa
un fallback runtime a tile di quattro colonne, senza un limite massimo su k.

`cuda_warp_smem` e' `cuda_warp` con **una sola variabile cambiata**: da dove
arrivano gli elementi di X. In `cuda_warp` ogni warp percorre tutta X, quindi
il traffico in lettura su X vale `m*n*k` elementi contro gli `m*n` di A: non
arriva in DRAM (X sta in L2) ma satura la banda L2, e cresce con k. Qui gli 8
warp di un blocco, che allo stesso passo `j` vogliono le stesse righe di X, ne
stagiano cooperativamente un tile `TJ x k` in shared memory e lo riusano: il
traffico verso L2 si divide per il numero di warp per blocco. Tutto il resto
(ordine dei cicli, lettura singola di A, k accumulatori in registro, riduzione
con `__shfl_down_sync`, i cinque k come template) resta identico, ed e' questo
che rende il confronto una misura e non un aneddoto.

La riga del tile in shared memory e' lunga `k + SMEM_PAD` scalari. Il motivo e'
il conflitto sui banchi: un double occupa due dei 32 banchi da 32 bit, la lane
`l` legge a stride `2*(k+SMEM_PAD)` banchi, e per `k=32` senza padding lo
stride e' 64 banchi, cioe' 0 (mod 32) — tutte e 32 le lane sullo stesso banco,
conflitto a 32 vie. E' l'analogo esatto del padding `LDA = N+24` gia' studiato
contro i conflict miss di L1 sulla CPU: stesso fenomeno, altra gerarchia. Il
padding e' un knob proprio perche' i due numeri vanno misurati entrambi, e i
due binari coesistono:

```bash
make KERNEL=cuda_warp_smem              # padding k+1, bin/...-cuda_warp_smem
make KERNEL=cuda_warp_smem SMEM_PAD=0   # bin/...-cuda_warp_smem-smempad0
mpirun -np 1 ./bin/matmul_mpi-cuda_warp_smem          -M 20000 -N 20000 -k 32 --reps 10
mpirun -np 1 ./bin/matmul_mpi-cuda_warp_smem-smempad0 -M 20000 -N 20000 -k 32 --reps 10
```

`TJ` non e' una costante: il tile pesa `TJ*(k+SMEM_PAD)*sizeof(scalar_t)` e k si
conosce solo a runtime, quindi TJ viene scelto dal budget di shared memory per
blocco (16 KiB) e la memoria e' allocata dinamicamente al lancio. Il budget non
e' il massimo di 48 KiB per una ragione di occupancy: con 64 KiB di shared per
SM, 16 KiB per blocco lasciano risiedere 4 blocchi (32 warp) per SM, 48 KiB ne
lascerebbero uno solo.

`cublas` non e' una proposta ma il **riferimento esterno**: serve a rispondere
con un numero misurato sulla stessa macchina e la stessa pipeline alla domanda
"perche' non avete chiamato una libreria". Entra nel progetto come un backend
qualsiasi, dietro la stessa interfaccia, cosi' che i tempi siano confrontabili
con gli altri. cuBLAS e' column-major e il progetto e' row-major, ma nessuna
trasposizione e' necessaria: una matrice `PxQ` row-major con leading dimension
`ld` e' gia', bit per bit, la `QxP` column-major con la stessa `ld`, quindi
`Y_cm = X_cm * A_cm` e la chiamata e' `cublas<t>gemm(h, N, N, k, m, n, ...)`
con `dX` e `dA` in quest'ordine. L'attesa e' che vada male, ed e' il risultato:
con `k <= 32` un GEMM general-purpose non ammortizza il proprio percorso.

Validazione completa e controllo dei registri sul server:

```bash
make KERNEL=cuda_warp check
make KERNEL=cuda_warp PREC=float check
make KERNEL=cuda_warp_smem check
make KERNEL=cuda_warp_smem PREC=float check
make KERNEL=cuda_warp_smem SMEM_PAD=0 check
make KERNEL=cublas check
make KERNEL=cublas PREC=float check
make -B KERNEL=cuda_warp EXTRA_NVCCFLAGS='-Xptxas -v'
```

Il benchmark di uscita va eseguito in double, su un solo rank/GPU, e confrontato
con `cuda_naive` sulla stessa configurazione. La metrica del backend e'
`GFLOPS kernel-only`:

```bash
mpirun -np 1 --bind-to core --map-by core ./bin/matmul_mpi-cuda_warp \
  -M 20000 -N 20000 -k 8 --pr 1 --pc 1 --a-mode local \
  --warmup 2 --reps 10
```

Le fasi cadono cosi':

| fase | dove | cronometrata? |
|---|---|---|
| creazione contesto CUDA, tutti i `cudaMalloc`, H2D di **A** | `local_gemm_create` | no, e' preprocessing (`t_setup_s`) |
| H2D di **X**, kernel, D2H di **Y** | `local_gemm` | si', e' `t_local` |
| solo il kernel, misurato con i `cudaEvent` | dentro `local_gemm` | si', ed e' `t_kernel` |

`t_local - t_kernel` viene riportato come `transfer/runtime overhead`: contiene
H2D e D2H, ma anche lancio, record/sync degli event, controlli CUDA e altro
overhead host. Non viene presentato impropriamente come misura del solo PCIe.

I buffer di X e Y vengono allocati in `local_gemm_create`, che riceve anche
`ldx` e `ldy`; una `cudaDeviceSynchronize` conclude realmente il preprocessing
prima di restituire. Di conseguenza nessuna allocazione o coda residua della
copia di A entra nella prima repetition, neppure con `--warmup 0`.

Ogni rank sceglie la GPU usando il local rank esportato da OpenMPI, MVAPICH o
Slurm; `SCPA_CUDA_DEVICE` resta disponibile come override esplicito. Se sul
nodo ci sono piu' rank locali che GPU, il programma avverte che le GPU vengono
condivise: il risultato resta corretto, ma i tempi `t_kernel` e ufficiali non
rappresentano il caso one-rank-per-GPU. Per misure confrontabili va quindi
usato al massimo un rank per GPU, salvo documentare esplicitamente la
condivisione (per esempio tramite MPS).

Se il blocco locale non entra in VRAM (40000x40000 in double sono 12.8 GiB
contro i 15.5 GiB della Quadro RTX 5000) il programma si ferma prima di
allocare, dicendo quanta memoria serve e quanta ce n'e'.

Per ogni repetition si prende separatamente il massimo fra i rank dei tempi di
broadcast, calcolo locale, reduce, totale end-to-end e tempo ufficiale. Il
totale end-to-end e' sempre misurato direttamente dall'inizio del broadcast
alla fine della reduce. Il tempo ufficiale locale vale:

```text
t_official = t_total                         backend CPU
t_official = t_bcast + t_kernel + t_reduce backend CUDA
```

La `MPI_Reduce(MPI_MAX)` viene applicata a `t_official` repetition per
repetition prima di calcolarne la media. Le metriche sono:

```text
gflops         = 2*M*N*k / mean(max_rank(t_official)) / 1e9 metrica ufficiale
gflops_compute = 2*M*N*k / mean(max_rank(t_compute))  / 1e9 solo calcolo
gflops_kernel  = 2*M*N*k / mean(max_rank(t_kernel))   / 1e9 alias CUDA
```

`t_compute` coincide con `t_local` su CPU e con `t_kernel` su CUDA. `t_total`
resta disponibile come misura end-to-end reale, comprensiva dei trasferimenti.

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
