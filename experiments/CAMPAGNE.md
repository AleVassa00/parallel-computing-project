# Campagne di misura C1 - C6

`guida_experimets.md` spiega **cosa significa ogni chiave** di un `.conf`.
Questo file spiega **quali campagne esistere, in che ordine eseguirle e cosa
leggere nei CSV che producono**.

Ogni campagna e' una *suite*: un file `.txt` che elenca i `.conf` da eseguire
in sequenza.

```bash
./script/run_cuda_experiments.sh --suite experiments/<campagna>.txt
```

Tutti i `.conf` di una stessa campagna condividono `outdir=`, quindi i loro
CSV finiscono nella **stessa cartella** con nomi diversi (uno per `name=`) e
non si sovrascrivono a vicenda.

---

## Ordine consigliato

| # | suite | cosa risponde | P | tempo indicativo |
|---|---|---|---|---|
| C3 | `c3_cpu_baseline.txt` | qual e' il denominatore di ogni speedup, e quanto vale la specializzazione su k | 1 | ~5 min |
| C2 | `c2_k_sweep.txt` | come si comporta ogni backend al variare di k, incluso il fallback generico | 1 | ~10 min |
| C1 | `c1_size_ratio_cpu.txt`<br>`c1_size_ratio_gpu.txt` | **campagna obbligatoria**: dimensioni crescenti su 3 rapporti M:N | 1 | ~5 + ~3 min |
| C6 | `c6_grid_shape.txt` | quale forma di griglia conviene a P fisso, cioe' perche' la griglia e' 2D | 8/16/20 | ~6 min |
| C4 | `c4_strong_scaling.txt` | come scala a problema fisso | 1..20 | ~5 min |
| C5 | `c5_weak_scaling.txt` | quanto costa la sola comunicazione | 1..20 | ~5 min |

Totale ~45 minuti di macchina, escluse le compilazioni. C3 e C2 vengono per
primi perche' producono i numeri di riferimento che tutte le altre citano.

C3, C2 (riga `scheme_a`), C4, C5 e C6 **non richiedono nvcc**: se la build
CUDA non funziona, cinque campagne su sei restano eseguibili.

---

## Le campagne

### C1 - dimensioni crescenti x rapporto M:N

`c1_size_ratio_cpu.txt`, `c1_size_ratio_gpu.txt` -> `results/c1_size_ratio_{cpu,gpu}/`

La traccia lo chiede esplicitamente: misure ripetute per M ed N crescenti, su
almeno 3 rapporti fra M ed N, e l'insieme deve contenere M = N.

Quattro passi (`s1`..`s4`, fattore 4 in `M*N` a ogni passo) per tre rapporti,
scelti in modo che a parita' di passo il lavoro sia lo stesso entro il 5%:

| passo | 1:1 | 3:1 (M=3N) | 1:2 (N=2M) | `M*N` |
|---|---|---|---|---|
| s1 | 3200 x 3200 | 5400 x 1800 | 2250 x 4500 | ~1.0e7 |
| s2 | 6400 x 6400 | 10800 x 3600 | 4500 x 9000 | ~4.0e7 |
| s3 | 12800 x 12800 | 21600 x 7200 | 9000 x 18000 | ~1.6e8 |
| s4 | 25600 x 25600 | 43200 x 14400 | 18000 x 36000 | ~6.4e8 |

Poiche' il lavoro e' uguale a parita' di passo, le tre curve si sovrappongono
sullo stesso asse e la differenza che si legge e' la **forma** del blocco, non
la taglia. Conta perche' il rapporto M:N decide quanto pesa il `Bcast` di X
(~`n_loc*k`) contro la `Reduce` di Y (~`m_loc*k`), e su GPU quanti warp
esistono (`m_loc`) contro quanto e' lungo il ciclo interno (`n_loc`): una
matrice larga e bassa affama la GPU di parallelismo.

A `s4` la sola A occupa ~5 GB in double: entra nella VRAM da 15.5 GiB della
Quadro RTX 5000, ma e' il primo punto che puo' fallire. Le suite sono in
ordine di taglia crescente e hanno `continue_on_error=1`, quindi un eventuale
OOM non invalida i punti gia' raccolti.

`reps=10` invece di 20: sono 24 punti per backend. La colonna
`t_official_cv_pct` dice riga per riga se la media e' stabile.

Per ripetere lo stesso sweep su un altro backend basta cambiare `kernel=` nei
12 `.conf` corrispondenti (e `name=`/`outdir=`, per non sovrascrivere).

### C2 - sweep su k, un backend per riga

`c2_k_sweep.txt` -> `results/c2_k_sweep/`

`ks=3 6 8 20 32 1 7 17 40 64`: prima i cinque k che la traccia impone, poi
cinque k **senza specializzazione**, che attraversano il fallback. La
differenza fra le due famiglie di righe e' la misura di quanto vale la
specializzazione. Su `scheme_a`, k=40 e k=64 costano anche una rilettura di A
(gli accumulatori sono bloccati a `KB=32`).

Sei backend: `scheme_a`, `cuda_naive`, `cuda_warp`, `cuda_warp_smem`,
`cuda_warp_tiled`, `cublas`. `cublas` non e' una proposta ma il **riferimento
esterno**: misurato sulla stessa macchina, con la stessa pipeline e lo stesso
cronometro, risponde con un numero a "perche' non avete chiamato una
libreria".

Su GPU la colonna da leggere e' `gflops_kernel`; su CPU e' `gflops`
(`t_kernel` vale -1, dove il kernel *e'* l'invocazione).

### C3 - baseline di CPU e specializzazione vs fallback

`c3_cpu_baseline.txt` -> `results/c3_cpu_baseline/`

Due ruoli in una campagna sola:

1. il **denominatore** di ogni speedup della relazione: il nucleo di calcolo a
   1 rank, senza comunicazione e senza GPU;
2. il microbenchmark che il README promette: `FORCE_GENERIC_K=0` contro
   `FORCE_GENERIC_K=1` sullo stesso problema. Le due build sono binari
   distinti (`matmul_mpi` e `matmul_mpi-generic`) con `kernel_name()` diverso,
   quindi le righe restano distinguibili nel CSV.

k=40 e k=64 sono un **controllo**: passano dal fallback in entrambe le build,
quindi le due righe devono coincidere. Se non coincidono, la differenza
misurata sugli altri k e' rumore, non specializzazione.

### C4 - strong scaling

`c4_strong_scaling.txt` -> `results/c4_strong_scaling/`

Problema fisso 20000 x 20000 (A da 3.2 GB), P = 1, 2, 4, 8, 16, 20 sulla forma
piu' quadrata - la stessa che `grid_default_shape` sceglierebbe da sola. A
P=20 il blocco locale e' 5000 x 4000, abbastanza da non stare in cache.

`ks=3 32` perche' calcolo e comunicazione crescono **entrambi** con k, ma da
basi diverse: con k=3 la scalabilita' e' quasi ideale, con k=32 il
`Bcast`/`Reduce` pesa dieci volte tanto ed e' li' che si rompe.

Colonne per il grafico: `gflops` piu' la terna `t_bcast_mean_s`,
`t_local_mean_s`, `t_reduce_mean_s` riportata come frazione di
`t_total_mean_s`.

### C5 - weak scaling

`c5_weak_scaling.txt` -> `results/c5_weak_scaling/`

Blocco locale **costante** a 5000 x 5000 su ogni processo: `M = 5000*pr`,
`N = 5000*pc`. Il lavoro per rank non cambia da P=1 a P=20, quindi il tempo
ideale e' piatto e tutto cio' che sale e' comunicazione pura. Separa calcolo e
comunicazione meglio dello strong scaling, dove i due effetti si sovrappongono.

Da dire in relazione: su una griglia 2D il weak scaling cambia anche il
rapporto M:N (M segue `pr`, N segue `pc`). E' inerente alla decomposizione, ed
e' il motivo per cui l'effetto della forma si misura separatamente in C1.

### C6 - forma della griglia a P fisso

`c6_grid_shape.txt` -> `results/c6_grid_shape/`

`all_grids=P` esegue **tutte** le fattorizzazioni ordinate:

- P=8: 1x8, 2x4, 4x2, 8x1
- P=16: 1x16, 2x8, 4x4, 8x2, 16x1
- P=20: 1x20, 2x10, 4x5, 5x4, 10x2, 20x1

E' la campagna che giustifica la griglia **2D** invece di una 1D, cioe' il
requisito centrale della traccia. Il volume comunicato dipende dalla forma in
due modi opposti: con `pr=1` non c'e' broadcast ma la reduce coinvolge tutti i
P processi, con `pc=1` e' il contrario. `t_bcast_mean_s` e `t_reduce_mean_s`
vanno letti **in coppia**: devono muoversi in direzioni opposte lungo la
sequenza di forme, e l'ottimo sta in mezzo. Le forme degenere 1xP e Px1 sono
le due decomposizioni monodimensionali, quindi il confronto e' diretto e sullo
stesso grafico.

---

## Unire i CSV di una campagna

Tutti i CSV di una cartella hanno la stessa intestazione (la emette il binario
stesso con `--csv-header`, quindi non puo' divergere dalle colonne):

```bash
find results/c1_size_ratio_cpu -name '*.csv' ! -name '*_raw.csv' ! -name '*_failures.csv' \
  | sort | xargs awk 'FNR==1 && NR!=1 {next} {print}' > results/c1_size_ratio_cpu/merged.csv
```

I file `*_raw.csv` hanno un'altra intestazione (una riga per repetition,
per lo studio della varianza) e vanno uniti separatamente.

---

## Prima di lanciare, sul server

- **Rilanciare una campagna sovrascrive i suoi CSV.** Il runner scrive
  l'intestazione con `>` e azzera il raw, ma *appende* al file dei fallimenti.
  Archiviare la cartella prima di ripetere:
  `mv results/c4_strong_scaling results/c4_strong_scaling.$(date +%F_%H%M)`.
- **Binding obbligatorio.** I `.conf` multi-rank impostano `bind_to=core` e
  `map_by=core`; senza, le misure sono rumore.
- **Trasporto MPI.** Il runner usa `btl=self,sm` per default. Il componente
  `sm` non esiste in Open MPI 3.x/4.x (li' si chiama `vader`): se le run
  multi-rank falliscono con un errore di BTL, verificare con
  `ompi_info --param btl all` e aggiungere `--btl self,vader` sulla riga di
  comando (i flag CLI hanno la precedenza sul `.conf`).
- **bash >= 4.4.** Il runner espande array potenzialmente vuoti sotto
  `set -u`; su bash 3.2 (macOS) si ferma con `unbound variable`. Sul server
  Linux non si presenta.
- **Validazione a parte.** Nessuna di queste campagne usa `--check`: la
  validazione costa `O(M*N*k)` seriale sul rank 0 e va fatta prima, con
  `make check`, `make PREC=float check` e `make check-padding` su taglie
  piccole.
