# Prodotto matrice x multivettore  -  Y = A*X
#
#   make            costruisce bin/matmul_mpi e bin/test_index
#   make test       esegue i test delle funzioni indice
#   make check      validazione MPI contro il seriale su piu' forme di griglia
#   make check-cxx  verifica che kernel.h resti utilizzabile da nvcc
#   make PREC=float ricompila in singola precisione
#   make KERNEL=..  seleziona l'implementazione di local_gemm (.c oppure .cu)
#   make FORCE_GENERIC_K=1  forza il fallback generico di scheme_a
#   make KERNEL=cuda_warp  backend CUDA warp-per-row con dispatch su k
#   make KERNEL=cuda_warp_smem  come sopra, ma con il tile di X in shared
#   make KERNEL=cuda_warp_smem SMEM_PAD=0  la stessa cosa senza il padding k+1
#   make KERNEL=cuda_warp_smem TILE_GRANULARITY=<n>   arrotondamento righe/tile
#   make KERNEL=cuda_warp_smem SMEM_BUDGET_BYTES=<n>  budget forzato invece che
#                                                     derivato dall'occupancy
#   make BLOCK=<n>  thread per blocco dei backend CUDA (default 256)
#   make KERNEL=cublas     riferimento esterno (aggiunge -lcublas da solo)
#   make KERNEL=omp_scheme_a        schema A parallelizzato con OpenMP
#   make KERNEL=omp_scheme_a_tiled  come sopra, con il tiling di X in cache
#   make KERNEL=omp_scheme_a_tiled OMP_X_TILE_BYTES=<n>  budget del tile di X
#   make KERNEL=omp_scheme_a_tiled OMP_Y_TILE_BYTES=<n>  budget del blocco di Y
#   make check-omp  validazione dei due backend OpenMP a 1 e 4 thread
#
# Un backend che include <omp.h> viene riconosciuto da solo e compilato con
# -fopenmp, esattamente come un .cu viene riconosciuto e compilato con nvcc:
# non c'e' nessun flag OPENMP=1 da ricordarsi di passare.
#
# Il backend si sceglie con KERNEL e il Makefile capisce da solo se e' un file C
# o un file CUDA:  src/kernel/$$(KERNEL).cu ha la precedenza su .c, viene
# compilato con nvcc e il binario viene linkato con -lcudart. Serve nvcc nel
# PATH (sul server:  module load cuda, oppure NVCC=/usr/local/cuda/bin/nvcc).
#
#   make KERNEL=cuda_naive check   ->   bin/matmul_mpi-cuda_naive, validato
#
# Ogni configurazione produce un BINARIO CON NOME PROPRIO: la configurazione di
# riferimento (double, scheme_a) e' bin/matmul_mpi, ogni variante aggiunge un
# suffisso (bin/matmul_mpi-float, bin/matmul_mpi-generic, ...). Cosi' un
# `make PREC=float` non puo' sovrascrivere il binario in doppia precisione e
# far misurare in float una campagna che si credeva in double.

CC     := gcc
CXX    := g++
MPICC  := mpicc
NVCC   ?= nvcc

# Turing (Quadro RTX 5000, sm_75) e' l'architettura del server. Compilare per
# la sola architettura di destinazione, e non per un fat binary, e' quello che
# serve: il codice non deve girare altrove.
NVCC_ARCH ?= sm_75

# Il kernel locale e' un file intercambiabile: il resto del codice vede solo
# l'interfaccia local_gemm dichiarata in src/kernel/kernel.h.
KERNEL ?= scheme_a
PREC   ?= double
FORCE_GENERIC_K ?= 0
TEST_A_PADDING  ?= 0

# Scalari di padding aggiunti a ogni riga del tile di X in shared memory dal
# backend cuda_warp_smem. 1 rompe il conflitto a 32 vie sui banchi che si
# presenta a k=32 in double; SMEM_PAD=0 e' il termine di paragone da misurare.
# Riguarda solo i backend CUDA che lo leggono, ma entra nel nome della
# configurazione: le due build coesistono come binari distinti.
SMEM_PAD        ?= 1

# Granularita' di arrotondamento delle righe per tile di cuda_warp_smem.
# 32 e' il default e NON e' un requisito di correttezza: il kernel gestisce gia'
# tile parziali. E' un'ottimizzazione ("nessuna lane inattiva nell'ultimo passo
# di warp") che a k=32 costa meta' del tile, e il cui bilancio va misurato.
TILE_GRANULARITY ?= 32

# Budget di shared memory per blocco di cuda_warp_smem, in byte.
# VUOTO = derivato a runtime dall'occupancy, che e' il default: il budget viene
# scelto in modo da non essere lui il vincolo attivo. Un valore esplicito lo
# forza, ed e' il termine di paragone dello sweep (p.es. SMEM_BUDGET_BYTES=16384
# riproduce il valore che prima era scritto a mano nel sorgente).
SMEM_BUDGET_BYTES ?=

ifeq ($(KERNEL),cuda_warp_smem)
ifeq ($(shell printf '%s\n' '$(TILE_GRANULARITY)' | grep -E '^[1-9][0-9]*$$'),)
$(error TILE_GRANULARITY deve essere un intero positivo)
endif
ifneq ($(SMEM_BUDGET_BYTES),)
ifeq ($(shell printf '%s\n' '$(SMEM_BUDGET_BYTES)' | grep -E '^[1-9][0-9]*$$'),)
$(error SMEM_BUDGET_BYTES deve essere un intero positivo, oppure vuoto per derivarlo)
endif
endif
endif

# Budget in byte dei due tile di cache di omp_scheme_a_tiled: quello di X, che
# deve stare in L1 dati, e quello di Y, che deve restare caldo in L2 per tutta
# la scansione di X. I default (16 KiB e 256 KiB) sono un'ipotesi sulla
# gerarchia del server, non una verita': sono knob perche' vanno misurati, e
# come SMEM_PAD entrano nel nome della configurazione, cosi' le build
# coesistono come binari distinti con un kernel_name() diverso ciascuna.
# Il NUMERO DI RIGHE per tile lo deriva il backend a runtime, perche' dipende
# da k: righe = budget / (k * sizeof(scalar_t)).
OMP_X_TILE_BYTES ?= 16384
OMP_Y_TILE_BYTES ?= 262144

ifeq ($(KERNEL),omp_scheme_a_tiled)
ifeq ($(shell printf '%s\n' '$(OMP_X_TILE_BYTES)' | grep -E '^[1-9][0-9]*$$'),)
$(error OMP_X_TILE_BYTES deve essere un intero positivo)
endif
ifeq ($(shell printf '%s\n' '$(OMP_Y_TILE_BYTES)' | grep -E '^[1-9][0-9]*$$'),)
$(error OMP_Y_TILE_BYTES deve essere un intero positivo)
endif
endif

ifeq ($(KERNEL),cuda_warp_tiled)
WARP_COL_TILE   ?= 8
ifeq ($(shell printf '%s\n' '$(WARP_COL_TILE)' | grep -E '^[1-9][0-9]*$$'),)
$(error WARP_COL_TILE deve essere un intero positivo)
endif
endif

# Thread per blocco dei kernel CUDA. 256 e' il default: 8 warp, e un divisore di
# 1024, che su Turing e' il massimo di thread residenti per SM. Uno sweep su
# 64/128/192/256/384/512/1024 misura quanto il kernel dipenda davvero da questo
# parametro, invece di lasciare la scelta come asserzione non verificata.
# Deve essere un multiplo di 32: i backend lo impongono con una #error.
# Come SMEM_PAD entra nel nome della configurazione, cosi' le build coesistono
# come binari distinti e il CSV riporta un kernel_name() diverso per ciascuna.
BLOCK           ?= 256
EXTRA_CFLAGS    ?=
EXTRA_NVCCFLAGS ?=

# -march=native su x86, -mcpu=native su aarch64: si prova cosa accetta il
# compilatore, cosi' lo stesso Makefile vale su VM di sviluppo e su server.
ARCHFLAGS := $(shell \
    $(CC) -march=native -E -x c /dev/null >/dev/null 2>&1 && echo -march=native || \
    ($(CC) -mcpu=native -E -x c /dev/null >/dev/null 2>&1 && echo -mcpu=native))

CFLAGS := -std=c11 -O3 $(ARCHFLAGS) -Wall -Wextra -Wpedantic -Isrc -MMD -MP \
	$(EXTRA_CFLAGS)

# Suffisso che identifica la configurazione. Vuoto per quella di riferimento,
# cosi' `make` continua a produrre bin/matmul_mpi e gli esempi del README
# restano validi; ogni scostamento dal default si porta dietro il proprio nome.
CONFIG :=
ifneq ($(PREC),double)
CONFIG := $(CONFIG)-$(PREC)
endif
ifneq ($(KERNEL),scheme_a)
CONFIG := $(CONFIG)-$(KERNEL)
endif
ifneq ($(FORCE_GENERIC_K),0)
CONFIG := $(CONFIG)-generic
endif
ifneq ($(TEST_A_PADDING),0)
CONFIG := $(CONFIG)-pad$(TEST_A_PADDING)
endif
ifneq ($(SMEM_PAD),1)
CONFIG := $(CONFIG)-smempad$(SMEM_PAD)
endif
ifneq ($(BLOCK),256)
CONFIG := $(CONFIG)-blk$(BLOCK)
endif
ifeq ($(KERNEL),cuda_warp_tiled)
ifneq ($(WARP_COL_TILE),8)
CONFIG := $(CONFIG)-tile$(WARP_COL_TILE)
endif
endif
# Solo cuda_warp_smem legge questi due: entrano nel nome della configurazione
# soltanto li', altrimenti si otterrebbero binari con nomi diversi e contenuto
# identico, indistinguibili nel CSV perche' kernel_name() non cambierebbe.
# Come sopra: solo omp_scheme_a_tiled legge questi due budget, quindi solo li'
# entrano nel nome, altrimenti si otterrebbero binari con nomi diversi e
# contenuto identico.
ifeq ($(KERNEL),omp_scheme_a_tiled)
ifneq ($(OMP_X_TILE_BYTES),16384)
CONFIG := $(CONFIG)-xtile$(OMP_X_TILE_BYTES)
endif
ifneq ($(OMP_Y_TILE_BYTES),262144)
CONFIG := $(CONFIG)-ytile$(OMP_Y_TILE_BYTES)
endif
endif
ifeq ($(KERNEL),cuda_warp_smem)
ifneq ($(TILE_GRANULARITY),32)
CONFIG := $(CONFIG)-g$(TILE_GRANULARITY)
endif
ifneq ($(SMEM_BUDGET_BYTES),)
CONFIG := $(CONFIG)-bud$(SMEM_BUDGET_BYTES)
endif
endif
LDLIBS := -lm

# Su nodo singolo la comunicazione deve passare da memoria condivisa. Su
# alcune configurazioni OpenMPI il componente TCP viene comunque provato e
# inonda stderr di avvisi: qui si impone il trasporto corretto.
# Sul server aggiungere il binding:  --bind-to core --map-by core
MPIFLAGS ?= --mca btl self,sm

ifeq ($(PREC),float)
PRECDEF := -DUSE_FLOAT
else
PRECDEF :=
endif
CFLAGS += $(PRECDEF)

ifneq ($(FORCE_GENERIC_K),0)
CFLAGS += -DFORCE_GENERIC_K
endif

ifneq ($(TEST_A_PADDING),0)
CFLAGS += -DTEST_A_PADDING=$(TEST_A_PADDING)
endif

# Tutto il progetto tranne il kernel: questi file sono C e non cambiano mai.
C_SRCS := \
	src/common/util.c \
	src/index/index.c \
	src/gen/gen.c \
	src/serial/serial.c \
	src/mpi/grid.c \
	src/mpi/distrib.c \
	src/mpi/matmul_mpi.c \
	src/bench/check.c \
	src/bench/main.c

# Il kernel invece puo' essere C o CUDA, e la differenza la decide l'estensione
# del file che esiste: .cu ha la precedenza. E' questo che rende il backend
# davvero intercambiabile - non c'e' un flag CUDA=1 da ricordarsi di passare.
KERNEL_SRC := $(firstword $(wildcard src/kernel/$(KERNEL).cu src/kernel/$(KERNEL).c))
ifeq ($(KERNEL_SRC),)
$(error KERNEL='$(KERNEL)': non esiste ne' src/kernel/$(KERNEL).cu ne' src/kernel/$(KERNEL).c)
endif

ifeq ($(suffix $(KERNEL_SRC)),.cu)
KERNEL_IS_CUDA := 1
else
KERNEL_IS_CUDA := 0
endif

# Stessa idea del riconoscimento .cu/.c e di quello di cuBLAS: un backend che
# include <omp.h> ha bisogno di -fopenmp per compilare le direttive e per
# linkare il runtime, e il Makefile lo deduce dal sorgente invece di chiedere
# un flag OPENMP=1 da ricordarsi. Senza -fopenmp le direttive sarebbero
# silenziosamente ignorate: si otterrebbe un binario che gira, che valida, e
# che misura un thread solo credendo di misurarne venti. E' esattamente il
# genere di errore che non lascia tracce, quindi il flag non puo' essere
# opzionale ne' dimenticabile.
KERNEL_IS_OPENMP := $(if $(shell grep -l '<omp\.h>' $(KERNEL_SRC) 2>/dev/null),1,0)

ifeq ($(KERNEL_IS_OPENMP),1)
ifeq ($(shell $(MPICC) -fopenmp -E -x c /dev/null >/dev/null 2>&1 && echo yes),)
$(error KERNEL='$(KERNEL)' e' un backend OpenMP, ma '$(MPICC)' non c'e' nel PATH oppure non accetta -fopenmp)
endif
# -fopenmp resta in CFLAGS e non in LDFLAGS: la regola di link passa CFLAGS al
# compilatore, che e' il modo corretto di tirarsi dietro il runtime OpenMP.
CFLAGS += -fopenmp
CFLAGS += -DSCPA_OMP_X_TILE_BYTES=$(OMP_X_TILE_BYTES) \
          -DSCPA_OMP_Y_TILE_BYTES=$(OMP_Y_TILE_BYTES)

# Il numero di thread e l'affinita' sono variabili d'ambiente, e mpirun non
# propaga automaticamente quelle che gli interessano: se sono definite qui,
# vanno inoltrate esplicitamente ai rank, altrimenti la validazione girerebbe
# sempre con il default del runtime e `make check-omp` non collauderebbe
# niente di diverso a ogni giro.
ifdef OMP_NUM_THREADS
MPIFLAGS += -x OMP_NUM_THREADS
endif
ifdef OMP_PROC_BIND
MPIFLAGS += -x OMP_PROC_BIND
endif
ifdef OMP_PLACES
MPIFLAGS += -x OMP_PLACES
endif
endif

ifeq ($(KERNEL_IS_CUDA),1)
# nvcc compila i .cu come C++: -std=c++14 riguarda il codice host del .cu, non
# il resto del progetto, che resta C11 compilato da mpicc. -march=native NON va
# dato a nvcc direttamente, che non lo conosce: passa al compilatore host con
# -Xcompiler. -lineinfo serve dopo, per correlare i profili di ncu al sorgente.
# -Xptxas -v stampa, per OGNI istanza template, registri usati e byte di spill.
# Non e' diagnostica occasionale: con acc[K] in registro il numero di registri
# per thread e' il vincolo che decide l'occupancy, e uno spill diverso da zero
# significa accumulatori finiti in local memory (cioe' in DRAM), che e' un
# problema che viene prima di qualunque discorso sul tiling. Tenerlo sempre
# acceso costa solo qualche riga a schermo e toglie la scusa di non guardare.
NVCCFLAGS := -O3 -std=c++14 -arch=$(NVCC_ARCH) -Isrc $(PRECDEF) -lineinfo \
	-DSCPA_SMEM_PAD=$(SMEM_PAD) -DSCPA_BLOCK_THREADS=$(BLOCK) \
	-Xptxas -v \
	-Xcompiler -Wall -Xcompiler -Wextra $(EXTRA_NVCCFLAGS)
ifeq ($(KERNEL),cuda_warp_tiled)
NVCCFLAGS += -DSCPA_WARP_COL_TILE=$(WARP_COL_TILE)
endif
ifeq ($(KERNEL),cuda_warp_smem)
NVCCFLAGS += -DSCPA_TILE_GRANULARITY=$(TILE_GRANULARITY)
ifneq ($(SMEM_BUDGET_BYTES),)
NVCCFLAGS += -DSCPA_SMEM_BUDGET_BYTES=$(SMEM_BUDGET_BYTES)
endif
endif
ifneq ($(ARCHFLAGS),)
NVCCFLAGS += -Xcompiler $(ARCHFLAGS)
endif

# Il link lo fa mpicc (servono le librerie MPI), quindi il runtime CUDA va
# aggiunto a mano; -lstdc++ perche' un oggetto prodotto da nvcc e' C++.
ifeq ($(origin CUDA_HOME), undefined)
CUDA_HOME := $(patsubst %/bin/,%,$(dir $(shell command -v $(NVCC) 2>/dev/null)))
endif
ifneq ($(CUDA_HOME),)
LDFLAGS += -L$(CUDA_HOME)/lib64
endif
LDLIBS += -lcudart -lstdc++

# Stessa idea del riconoscimento .cu/.c: un backend che include cublas_v2.h ha
# bisogno anche di -lcublas al link, e il Makefile lo deduce dal sorgente
# invece di chiedere un flag CUBLAS=1 da ricordarsi.
ifneq ($(shell grep -l cublas_v2.h $(KERNEL_SRC) 2>/dev/null),)
LDLIBS += -lcublas
endif
endif

# Configurazioni diverse non condividono ne' oggetti ne' binario: cambiare
# precisione, padding o dispatch non puo' quindi riutilizzare accidentalmente
# una vecchia build ne' sovrascrivere quella di riferimento.
OBJDIR ?= obj/matmul_mpi$(CONFIG)
KERNEL_OBJ := $(OBJDIR)/kernel/$(KERNEL).o
OBJS   := $(patsubst src/%.c,$(OBJDIR)/%.o,$(C_SRCS)) $(KERNEL_OBJ)
DEPS   := $(patsubst src/%.c,$(OBJDIR)/%.d,$(C_SRCS))

BIN ?= bin/matmul_mpi$(CONFIG)
TESTBIN := bin/test_index

.PHONY: all test check check-mpi check-cxx check-omp check-padding padding-run clean

all: $(BIN) $(TESTBIN)
	@echo "built $(BIN)  [PREC=$(PREC) KERNEL=$(KERNEL) ($(KERNEL_SRC)) FORCE_GENERIC_K=$(FORCE_GENERIC_K) TEST_A_PADDING=$(TEST_A_PADDING) SMEM_PAD=$(SMEM_PAD) BLOCK=$(BLOCK) TILE_GRANULARITY=$(TILE_GRANULARITY) SMEM_BUDGET_BYTES=$(if $(SMEM_BUDGET_BYTES),$(SMEM_BUDGET_BYTES),derivato)]"

$(OBJDIR)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -c $< -o $@

# Un kernel .cu lo compila nvcc. Nessuna generazione automatica delle
# dipendenze qui (l'opzione cambia fra le versioni di nvcc): gli header del
# kernel sono tre e si elencano sotto, esplicitamente.
$(OBJDIR)/kernel/%.o: src/kernel/%.cu
	@mkdir -p $(dir $@)
	@command -v $(NVCC) >/dev/null 2>&1 || { \
		echo "errore: '$(NVCC)' non trovato nel PATH."; \
		echo "        Il backend '$*' e' CUDA: va compilato sul server."; \
		echo "        Prova 'module load cuda' oppure NVCC=/usr/local/cuda/bin/nvcc."; \
		exit 1; }
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

ifeq ($(KERNEL_IS_CUDA),1)
$(KERNEL_OBJ): src/kernel/kernel.h src/common/scalar.h src/common/util.h
endif

# Niente target FORCE: ora che il nome del binario dipende dalla
# configurazione, la normale logica di dipendenza di make e' corretta.
$(BIN): $(OBJS)
	@mkdir -p bin
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# I test degli indici non usano ne' MPI ne' CUDA: compilatore normale, binario
# autonomo, e -lm scritto qui invece di $(LDLIBS), che con un backend .cu
# conterrebbe anche -lcudart.
$(TESTBIN): test/test_index.c src/index/index.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -o $@ -lm

test: $(TESTBIN)
	./$(TESTBIN)

# Validazione delle modalita' indipendenti di A e X, forme 1x1/1x4/4x1/2x2,
# k specializzati, fallback generico e blocchi vuoti.
# Eseguire anche `make PREC=float check`.
check: check-cxx check-mpi

check-mpi: all
	./$(TESTBIN)
	mpirun -np 1 $(MPIFLAGS) ./$(BIN) -M 257 -N 257 -k 3  --pr 1 --pc 1 --a-mode local  --x-mode local  --check --reps 2
	mpirun -np 1 $(MPIFLAGS) ./$(BIN) -M 257 -N 257 -k 3  --pr 1 --pc 1 --a-mode global --x-mode local  --check --reps 2
	mpirun -np 1 $(MPIFLAGS) ./$(BIN) -M 257 -N 257 -k 3  --pr 1 --pc 1 --a-mode local  --x-mode global --check --reps 2
	mpirun -np 1 $(MPIFLAGS) ./$(BIN) -M 257 -N 257 -k 3  --pr 1 --pc 1 --a-mode global --x-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 6  --pr 1 --pc 4 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 6  --pr 1 --pc 4 --a-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 6  --pr 1 --pc 4 --a-mode local  --x-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 8  --pr 4 --pc 1 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 8  --pr 4 --pc 1 --a-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 8  --pr 4 --pc 1 --a-mode local  --x-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 20 --pr 2 --pc 2 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 20 --pr 2 --pc 2 --a-mode global --x-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 64  -N 512 -k 32 --pr 2 --pc 2 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 64  -N 512 -k 32 --pr 2 --pc 2 --a-mode global --check --reps 2
	# k non specializzati: devono attraversare il fallback generico.
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 37 -N 29 -k 1  --pr 2 --pc 2 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 37 -N 29 -k 1  --pr 2 --pc 2 --a-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 37 -N 29 -k 7  --pr 2 --pc 2 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 37 -N 29 -k 7  --pr 2 --pc 2 --a-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 37 -N 29 -k 17 --pr 2 --pc 2 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 37 -N 29 -k 17 --pr 2 --pc 2 --a-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 37 -N 29 -k 40 --pr 2 --pc 2 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 37 -N 29 -k 40 --pr 2 --pc 2 --a-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 37 -N 29 -k 64 --pr 2 --pc 2 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 37 -N 29 -k 64 --pr 2 --pc 2 --a-mode global --check --reps 2
	# Blocchi vuoti sulle righe e sulle colonne della griglia.
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 2 -N 3 -k 7 --pr 4 --pc 1 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 2 -N 3 -k 7 --pr 4 --pc 1 --a-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 3 -N 2 -k 7 --pr 1 --pc 4 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 3 -N 2 -k 7 --pr 1 --pc 4 --a-mode global --x-mode global --check --reps 2

# kernel.h e' l'interfaccia che il backend CUDA dovra' implementare, e nvcc
# compila i .cu come C++. Questo target verifica in anticipo, senza bisogno di
# avere CUDA installato, che l'header resti parsabile da C++ e che i simboli
# conservino il linkage C.
check-cxx:
	@mkdir -p $(OBJDIR)
	$(CXX) -std=c++14 $(PRECDEF) -Isrc -Wall -Wextra \
		-c test/cxx_iface.cpp -o $(OBJDIR)/cxx_iface.o
	@if command -v nm >/dev/null 2>&1; then \
		for sym in local_gemm_create local_gemm local_gemm_destroy \
		           local_gemm_last_compute_seconds local_gemm_setup_seconds \
		           local_gemm_blocks_per_sm local_gemm_x_rows_per_tile \
		           local_gemm_threads \
		           kernel_name xmalloc die; do \
			nm -u $(OBJDIR)/cxx_iface.o | grep -qw $$sym || { \
				echo "check-cxx: FAIL: '$$sym' e' decorato (manca extern \"C\" in un header)"; \
				exit 1; }; \
		done; \
	fi
	@echo "check-cxx: kernel.h e util.h sono compilabili da C++ con linkage C (pronti per nvcc)"

# Validazione dei backend OpenMP.
#
# Il punto non e' rieseguire la stessa suite con un altro kernel: e' eseguirla
# con NUMERI DI THREAD DIVERSI. Il risultato del prodotto non dipende da quanti
# thread lo calcolano, quindi una corsa fra thread - una riga di Y scritta da
# due team, un intervallo che si sovrappone al vicino - si manifesta come un
# FAIL solo quando i thread sono piu' di uno, e non si manifesta affatto se si
# collauda alla cieca con il default del runtime.
#
# L'ultimo giro forza tile di 64 byte, cioe' pochissime righe per tile: e' il
# caso limite del percorso in accumulo, dove il ciclo su j viene spezzato
# centinaia di volte per riga e ogni confine di tile puo' sbagliare. Con i
# budget di default, sui k della traccia, quel percorso non verrebbe quasi mai
# attraversato.
check-omp:
	@for t in 1 2 4; do \
		for kern in omp_scheme_a omp_scheme_a_tiled; do \
			echo "== check-omp: KERNEL=$$kern OMP_NUM_THREADS=$$t"; \
			OMP_NUM_THREADS=$$t $(MAKE) --no-print-directory KERNEL=$$kern check-mpi || exit 1; \
		done; \
	done
	@echo "== check-omp: tile minimi (percorso in accumulo sotto stress)"
	OMP_NUM_THREADS=4 $(MAKE) --no-print-directory KERNEL=omp_scheme_a_tiled \
		OMP_X_TILE_BYTES=64 OMP_Y_TILE_BYTES=64 check-mpi
	@echo "check-omp: i due backend OpenMP validano a 1, 2 e 4 thread"

# Build isolata che forza lda=n_loc+8; non modifica la build normale.
check-padding:
	$(MAKE) TEST_A_PADDING=8 padding-run

padding-run: $(BIN)
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 8 \
		--pr 2 --pc 2 --a-mode global --check --reps 2

clean:
	rm -rf obj bin

-include $(DEPS)
