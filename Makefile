# Prodotto matrice x multivettore  -  Y = A*X
#
#   make            costruisce bin/matmul_mpi e bin/test_index
#   make test       esegue i test delle funzioni indice
#   make check      validazione MPI contro il seriale su piu' forme di griglia
#   make check-cxx  verifica che kernel.h resti utilizzabile da nvcc
#   make PREC=float ricompila in singola precisione
#   make KERNEL=..  seleziona l'implementazione di local_gemm
#   make FORCE_GENERIC_K=1  forza il fallback generico di scheme_a
#
# Ogni configurazione produce un BINARIO CON NOME PROPRIO: la configurazione di
# riferimento (double, scheme_a) e' bin/matmul_mpi, ogni variante aggiunge un
# suffisso (bin/matmul_mpi-float, bin/matmul_mpi-generic, ...). Cosi' un
# `make PREC=float` non puo' sovrascrivere il binario in doppia precisione e
# far misurare in float una campagna che si credeva in double.

CC     := gcc
CXX    := g++
MPICC  := mpicc

# Il kernel locale e' un file intercambiabile: il resto del codice vede solo
# l'interfaccia local_gemm dichiarata in src/kernel/kernel.h.
KERNEL ?= scheme_a
PREC   ?= double
FORCE_GENERIC_K ?= 0
TEST_A_PADDING  ?= 0
EXTRA_CFLAGS    ?=

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

SRCS := \
	src/common/util.c \
	src/index/index.c \
	src/gen/gen.c \
	src/serial/serial.c \
	src/kernel/$(KERNEL).c \
	src/mpi/grid.c \
	src/mpi/distrib.c \
	src/mpi/matmul_mpi.c \
	src/bench/check.c \
	src/bench/main.c

# Configurazioni diverse non condividono ne' oggetti ne' binario: cambiare
# precisione, padding o dispatch non puo' quindi riutilizzare accidentalmente
# una vecchia build ne' sovrascrivere quella di riferimento.
OBJDIR ?= obj/matmul_mpi$(CONFIG)
OBJS   := $(patsubst src/%.c,$(OBJDIR)/%.o,$(SRCS))
DEPS   := $(OBJS:.o=.d)

BIN ?= bin/matmul_mpi$(CONFIG)
TESTBIN := bin/test_index

.PHONY: all test check check-mpi check-cxx check-padding padding-run clean

all: $(BIN) $(TESTBIN)
	@echo "built $(BIN)  [PREC=$(PREC) KERNEL=$(KERNEL) FORCE_GENERIC_K=$(FORCE_GENERIC_K) TEST_A_PADDING=$(TEST_A_PADDING)]"

$(OBJDIR)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -c $< -o $@

# Niente target FORCE: ora che il nome del binario dipende dalla
# configurazione, la normale logica di dipendenza di make e' corretta.
$(BIN): $(OBJS)
	@mkdir -p bin
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# I test degli indici non usano MPI: compilatore normale, binario autonomo.
$(TESTBIN): test/test_index.c src/index/index.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test: $(TESTBIN)
	./$(TESTBIN)

# Validazione delle due modalita' di A, forme 1x1/1x4/4x1/2x2, k specializzati,
# fallback generico e blocchi vuoti. Eseguire anche `make PREC=float check`.
check: check-cxx check-mpi

check-mpi: all
	./$(TESTBIN)
	mpirun -np 1 $(MPIFLAGS) ./$(BIN) -M 257 -N 257 -k 3  --pr 1 --pc 1 --a-mode local  --check --reps 2
	mpirun -np 1 $(MPIFLAGS) ./$(BIN) -M 257 -N 257 -k 3  --pr 1 --pc 1 --a-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 6  --pr 1 --pc 4 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 6  --pr 1 --pc 4 --a-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 8  --pr 4 --pc 1 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 8  --pr 4 --pc 1 --a-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 20 --pr 2 --pc 2 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 20 --pr 2 --pc 2 --a-mode global --check --reps 2
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
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 3 -N 2 -k 7 --pr 1 --pc 4 --a-mode global --check --reps 2

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
