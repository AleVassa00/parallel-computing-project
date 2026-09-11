# Prodotto matrice x multivettore  -  Y = A*X
#
#   make            costruisce bin/matmul_mpi e bin/test_index
#   make test       esegue i test delle funzioni indice
#   make check      validazione MPI contro il seriale su piu' forme di griglia
#   make check-cxx  verifica che kernel.h resti utilizzabile da nvcc
#   make PREC=float ricompila in singola precisione
#   make KERNEL=..  seleziona l'implementazione di local_gemm (.c oppure .cu)
#   make FORCE_GENERIC_K=1  forza il fallback generico di scheme_a
#   make KERNEL=scheme_a_jblock     schema A con X mattonellata in cache
#   make KERNEL=scheme_a_jblock_rb  come sopra + register blocking sulle righe
#   make KERNEL=scheme_a_jblock JBLOCK_BYTES=<n>  byte di X per tile (default 65536)
#   make KERNEL=scheme_a_jblock_rb RB_ROWS=<1|2|4> forza le righe per blocco
#   make KERNEL=cuda_warp  backend CUDA warp-per-row con dispatch su k
#   make KERNEL=cuda_warp X_LAYOUT=column  X compatta per colonne (default row)
#   make KERNEL=cuda_warp_smem  come sopra, ma con il tile di X in shared
#   make KERNEL=cuda_warp_smem SMEM_PAD=0  la stessa cosa senza il padding k+1
#   make KERNEL=cuda_warp_smem TILE_GRANULARITY=<n>   arrotondamento righe/tile
#   make KERNEL=cuda_warp_smem SMEM_BUDGET_BYTES=<n>  budget forzato invece che
#                                                     derivato dall'occupancy
#   make BLOCK=<n>  thread per blocco dei backend CUDA (default 256)
#   make KERNEL=cublas     riferimento esterno (aggiunge -lcublas da solo)
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

# Layout del solo blocco locale di X, convertito dal driver in preprocessing.
# Le due build coesistono; i backend non abilitati rifiutano column.
X_LAYOUT ?= row
ifneq ($(X_LAYOUT),row)
ifneq ($(X_LAYOUT),column)
$(error X_LAYOUT deve essere row oppure column)
endif
endif
ifeq ($(X_LAYOUT),column)
ifneq ($(KERNEL),cuda_warp)
$(error X_LAYOUT=column e' supportato soltanto da cuda_warp)
endif
X_LAYOUT_DEF := -DX_COLUMN_MAJOR=1
endif

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

# Tile di X in cache dei backend scheme_a_jblock*: byte per tile. Il default
# sta nella L2 di qualunque core; e' il punto di partenza dello sweep.
# RB_ROWS forza le righe di A per blocco di scheme_a_jblock_rb; vuoto = scelto
# a compile-time dal register file dell'architettura (vedi il sorgente).
JBLOCK_BYTES ?= 65536
RB_ROWS      ?=
JBLOCK_KERNELS := scheme_a_jblock scheme_a_jblock_rb
ifneq ($(filter $(KERNEL),$(JBLOCK_KERNELS)),)
ifeq ($(shell printf '%s\n' '$(JBLOCK_BYTES)' | grep -E '^[1-9][0-9]*$$'),)
$(error JBLOCK_BYTES deve essere un intero positivo)
endif
endif
ifeq ($(KERNEL),scheme_a_jblock_rb)
ifneq ($(RB_ROWS),)
ifeq ($(filter $(RB_ROWS),1 2 4),)
$(error RB_ROWS deve valere 1, 2 oppure 4, oppure vuoto per derivarlo)
endif
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

# -ffp-contract=fast: con -std=c11 (ISO stretto) gcc disabilita la fusione di
# a*x+acc in una FMA e il kernel esegue mul e add separate, meta' dei FLOP
# per istruzione. Il flag ripristina il default di -std=gnu11 senza rinunciare
# alla pedanteria del resto.
CFLAGS := -std=c11 -O3 $(ARCHFLAGS) -ffp-contract=fast \
	-Wall -Wextra -Wpedantic -Isrc -MMD -MP \
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
ifeq ($(KERNEL),cuda_warp_smem)
ifneq ($(TILE_GRANULARITY),32)
CONFIG := $(CONFIG)-g$(TILE_GRANULARITY)
endif
ifneq ($(SMEM_BUDGET_BYTES),)
CONFIG := $(CONFIG)-bud$(SMEM_BUDGET_BYTES)
endif
endif
ifeq ($(X_LAYOUT),column)
CONFIG := $(CONFIG)-xcol
endif
ifneq ($(filter $(KERNEL),$(JBLOCK_KERNELS)),)
ifneq ($(JBLOCK_BYTES),65536)
CONFIG := $(CONFIG)-jb$(JBLOCK_BYTES)
endif
endif
ifeq ($(KERNEL),scheme_a_jblock_rb)
ifneq ($(RB_ROWS),)
CONFIG := $(CONFIG)-rb$(RB_ROWS)
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
CFLAGS += $(PRECDEF) $(X_LAYOUT_DEF)

ifneq ($(FORCE_GENERIC_K),0)
CFLAGS += -DFORCE_GENERIC_K
endif

ifneq ($(TEST_A_PADDING),0)
CFLAGS += -DTEST_A_PADDING=$(TEST_A_PADDING)
endif

ifneq ($(filter $(KERNEL),$(JBLOCK_KERNELS)),)
CFLAGS += -DJBLOCK_BYTES=$(JBLOCK_BYTES)
ifneq ($(RB_ROWS),)
CFLAGS += -DRB_ROWS=$(RB_ROWS)
endif
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
	$(X_LAYOUT_DEF) \
	-DSMEM_PAD=$(SMEM_PAD) -DBLOCK_THREADS=$(BLOCK) \
	-Xptxas -v \
	-Xcompiler -Wall -Xcompiler -Wextra $(EXTRA_NVCCFLAGS)
ifeq ($(KERNEL),cuda_warp_tiled)
NVCCFLAGS += -DWARP_COL_TILE=$(WARP_COL_TILE)
endif
ifeq ($(KERNEL),cuda_warp_smem)
NVCCFLAGS += -DTILE_GRANULARITY=$(TILE_GRANULARITY)
ifneq ($(SMEM_BUDGET_BYTES),)
NVCCFLAGS += -DSMEM_BUDGET_BYTES=$(SMEM_BUDGET_BYTES)
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

.PHONY: all test check check-mpi check-cxx check-padding padding-run clean

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
		           local_gemm_last_h2d_X_seconds local_gemm_last_d2h_Y_seconds \
		           local_gemm_bytes_h2d_A local_gemm_bytes_h2d_X_per_call \
		           local_gemm_bytes_d2h_Y_per_call \
		           local_gemm_setup_device_init_seconds \
		           local_gemm_setup_device_alloc_seconds \
		           local_gemm_setup_h2d_A_seconds \
		           local_gemm_blocks_per_sm local_gemm_x_rows_per_tile \
		           kernel_name xmalloc die; do \
			nm -u $(OBJDIR)/cxx_iface.o | grep -qw $$sym || { \
				echo "check-cxx: FAIL: '$$sym' e' decorato (manca extern \"C\" in un header)"; \
				exit 1; }; \
		done; \
	fi
	@echo "check-cxx: kernel.h e util.h sono compilabili da C++ con linkage C (pronti per nvcc)"

# Build isolata che forza lda=n_loc+8; non modifica la build normale.
check-padding:
	$(MAKE) TEST_A_PADDING=8 padding-run

padding-run: $(BIN)
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 8 \
		--pr 2 --pc 2 --a-mode global --check --reps 2

clean:
	rm -rf obj bin

-include $(DEPS)
