# Prodotto matrice x multivettore  -  Y = A*X
#
#   make            costruisce bin/matmul_mpi e bin/test_index
#   make test       esegue i test delle funzioni indice
#   make check      validazione MPI contro il seriale su piu' forme di griglia
#   make PREC=float ricompila in singola precisione
#   make KERNEL=..  seleziona l'implementazione di local_gemm
#   make FORCE_GENERIC_K=1  forza il fallback generico di scheme_a

CC     := gcc
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
LDLIBS := -lm

# Su nodo singolo la comunicazione deve passare da memoria condivisa. Su
# alcune configurazioni OpenMPI il componente TCP viene comunque provato e
# inonda stderr di avvisi: qui si impone il trasporto corretto.
# Sul server aggiungere il binding:  --bind-to core --map-by core
MPIFLAGS ?= --mca btl self,sm

ifeq ($(PREC),float)
CFLAGS += -DUSE_FLOAT
endif

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

# Configurazioni diverse non condividono oggetti: cambiare precisione, padding
# o dispatch non puo' quindi riutilizzare accidentalmente una vecchia build.
OBJDIR ?= obj/$(PREC)/$(KERNEL)/generic$(FORCE_GENERIC_K)-pad$(TEST_A_PADDING)
OBJS   := $(patsubst src/%.c,$(OBJDIR)/%.o,$(SRCS))
DEPS   := $(OBJS:.o=.d)

BIN ?= bin/matmul_mpi
TESTBIN := bin/test_index

.PHONY: all test check check-mpi check-padding padding-run clean FORCE

all: $(BIN) $(TESTBIN)

$(OBJDIR)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJS) FORCE
	@mkdir -p bin
	$(MPICC) $(CFLAGS) $(filter %.o,$^) -o $@ $(LDLIBS)

FORCE:

# I test degli indici non usano MPI: compilatore normale, binario autonomo.
$(TESTBIN): test/test_index.c src/index/index.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test: $(TESTBIN)
	./$(TESTBIN)

# Validazione delle due modalita' di A, forme 1x1/1x4/4x1/2x2, k specializzati,
# fallback generico e blocchi vuoti. Eseguire anche `make PREC=float check`.
check: check-mpi

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

# Build isolata che forza lda=n_loc+8; non modifica la build normale.
check-padding:
	$(MAKE) TEST_A_PADDING=8 \
		OBJDIR=obj/padding BIN=bin/matmul_mpi_padding padding-run

padding-run: $(BIN)
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 8 \
		--pr 2 --pc 2 --a-mode global --check --reps 2

clean:
	rm -rf obj bin

-include $(DEPS)
