```ini
# ============================================================
# IDENTIFICAZIONE DELL'ESPERIMENTO
# ============================================================

# Nome logico dell'esperimento.
# Verra' usato anche per organizzare i risultati.
name=naive_block_sweep

# Tipo di esperimento.
# Valori principali:
#   k-sweep
#   block-sweep
#   grid-sweep
#   compare
#   registers
#   smem-pad-sweep
#   ncu
#   full
experiment=block-sweep


# ============================================================
# KERNEL
# ============================================================

# Kernel da testare.
# Esempi:
#   cuda_naive
#   cuda_warp
#   cuda_warp_smem
#   cublas
#   scheme_a
kernel=cuda_naive

# Questa opzione serve soprattutto con experiment=compare
# o con alcuni esperimenti multipli.
# Se usi solo kernel=..., puoi ometterla.
#
# kernels=cuda_naive cuda_warp cuda_warp_smem


# ============================================================
# DIMENSIONI DEL PROBLEMA
# ============================================================

# Numero di righe della matrice A.
M=10000

# Numero di colonne della matrice A.
# Corrisponde anche al numero di righe di X.
N=10000


# ============================================================
# VALORI DI k
# ============================================================

# "k" indica un singolo valore.
# Usalo quando vuoi eseguire una sola configurazione.
#
# k=32

# "ks" indica invece una lista di valori.
# In un block-sweep ogni BLOCK verra' provato per ogni k.
ks=3 6 8 20 32


# ============================================================
# BLOCK SIZE CUDA
# ============================================================

# "block" indica un singolo block size.
# Usalo quando il block size deve rimanere fisso.
#
# block=256

# "blocks" indica la lista dei block size da provare.
#
# Devono essere multipli di 32 perche' il warp contiene 32 thread.
# Nel nostro progetto questi sono i valori principali che vogliamo testare.
blocks=64 128 192 256 384 512 1024


# ============================================================
# RIPETIZIONI DEL BENCHMARK
# ============================================================

# Numero di esecuzioni NON misurate prima delle vere misure.
# Servono a stabilizzare runtime/cache/clock GPU.
warmup=5

# Numero di esecuzioni utilizzate per calcolare le statistiche.
reps=20


# ============================================================
# SEED
# ============================================================

# Seed utilizzato per generare A e X.
# Specificarlo rende esplicita e riproducibile la campagna.
#
# Se omesso, viene usato il seed di default del programma.
seed=20252026


# ============================================================
# VALIDAZIONE
# ============================================================

# true:
#   dopo il benchmark viene effettuato il controllo rispetto
#   al riferimento seriale.
#
# false:
#   nessuna validazione.
#
# Per campagne prestazionali grandi normalmente puoi tenerlo false,
# dopo aver gia' validato il kernel separatamente.
check=false


# ============================================================
# GENERAZIONE / DISTRIBUZIONE DI A E X
# ============================================================

# local:
#   ogni processo genera direttamente il proprio blocco locale di A.
#
# global:
#   A viene inizialmente materializzata globalmente e poi distribuita.
a_mode=local

# local:
#   i processi interessati generano localmente la propria fetta di X.
#
# global:
#   X viene materializzata sul root e distribuita.
x_mode=local


# ============================================================
# PRECISIONE
# ============================================================

# Possibili valori:
#   double
#   float
#
# Nel progetto principale stiamo lavorando in double.
prec=double


# ============================================================
# MPI - GRIGLIA FISSA
# ============================================================

# Numero totale di processi MPI.
np=1

# Numero di righe della griglia MPI.
pr=1

# Numero di colonne della griglia MPI.
pc=1

# Deve sempre valere:
#
#   np = pr * pc
#
# Esempi:
#
#   np=4
#   pr=2
#   pc=2
#
# oppure:
#
#   np=4
#   pr=1
#   pc=4


# ============================================================
# MPI - TUTTE LE POSSIBILI GRIGLIE
# ============================================================

# In alternativa a np/pr/pc puoi usare:
#
# all_grids=4
#
# e lo script eseguira':
#
#   1x4
#   2x2
#   4x1
#
# Con:
#
# all_grids=8
#
# eseguira':
#
#   1x8
#   2x4
#   4x2
#   8x1
#
# NON usare normalmente all_grids insieme a una griglia fissa.
#
# all_grids=4


# ============================================================
# OPENMPI / CPU BINDING
# ============================================================

# Trasporto OpenMPI usato sul nodo singolo.
btl=self,sm

# Binding dei processi MPI ai core.
bind_to=core

# Strategia di mapping.
map_by=core

# Se true, OpenMPI stampa dove sono stati bindati i processi.
# Utile per controllare la configurazione, ma rumoroso nei benchmark normali.
report_bindings=false


# ============================================================
# SHARED MEMORY PADDING
# ============================================================

# Padding delle righe del tile shared di X.
#
# Ha senso soprattutto per cuda_warp_smem.
# Per cuda_naive non e' un parametro interessante.
smem_pad=1

# Per un esperimento smem-pad-sweep puoi invece usare:
#
# smem_pads=0 1


# ============================================================
# GENERIC K
# ============================================================

# 0:
#   usa normalmente le specializzazioni disponibili.
#
# 1:
#   forza il percorso generico nei kernel che lo supportano.
#
# Per cuda_naive normalmente non serve modificarlo.
force_generic_k=0


# ============================================================
# PADDING DI A
# ============================================================

# Padding aggiuntivo alla leading dimension di A.
#
# Serve solo per esperimenti specifici sul layout/accesso alla memoria.
# Nei benchmark normali lascia 0.
test_a_padding=0


# ============================================================
# ARCHITETTURA CUDA
# ============================================================

# Quadro RTX 5000 = Turing Compute Capability 7.5.
nvcc_arch=sm_75


# ============================================================
# NSIGHT COMPUTE
# ============================================================

# Set di metriche da usare con experiment=ncu.
#
# Non viene usato durante un normale block-sweep.
ncu_set=full


# ============================================================
# GESTIONE ERRORI
# ============================================================

# false:
#   se una configurazione fallisce, la campagna si interrompe.
#
# true:
#   registra il fallimento e passa alla configurazione successiva.
continue_on_error=false


# ============================================================
# OUTPUT A TERMINALE
# ============================================================

# false:
#   salva il CSV senza stampare tutte le righe di benchmark.
#
# true:
#   mostra anche le righe CSV durante l'esecuzione.
verbose=false


# ============================================================
# DIRECTORY OUTPUT
# ============================================================

# Normalmente NON serve specificarla.
#
# Con:
#   name=naive_block_sweep
#
# lo script crea automaticamente qualcosa tipo:
#
#   results/naive_block_sweep/20260908_230000/
#
# Se vuoi imporre manualmente una directory puoi invece usare:
#
# outdir=results/mia_cartella
```
