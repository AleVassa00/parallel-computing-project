# Prodotto matrice x multivettore  -  Y = A*X
#
#   make            costruisce bin/matmul_mpi e bin/test_index
#   make test       esegue i test delle funzioni indice
#   make check      validazione MPI contro il seriale su piu' forme di griglia
#   make PREC=float ricompila in singola precisione
#   make KERNEL=..  seleziona l'implementazione di local_gemm

CC     := gcc
MPICC  := mpicc

# Il kernel locale e' un file intercambiabile: il resto del codice vede solo
# l'interfaccia local_gemm dichiarata in src/kernel/kernel.h.
KERNEL ?= scheme_a
PREC   ?= double

# -march=native su x86, -mcpu=native su aarch64: si prova cosa accetta il
# compilatore, cosi' lo stesso Makefile vale su VM di sviluppo e su server.
ARCHFLAGS := $(shell \
    $(CC) -march=native -E -x c /dev/null >/dev/null 2>&1 && echo -march=native || \
    ($(CC) -mcpu=native -E -x c /dev/null >/dev/null 2>&1 && echo -mcpu=native))

CFLAGS := -std=c11 -O3 $(ARCHFLAGS) -Wall -Wextra -Wpedantic -Isrc -MMD -MP
LDLIBS := -lm

# Su nodo singolo la comunicazione deve passare da memoria condivisa. Su
# alcune configurazioni OpenMPI il componente TCP viene comunque provato e
# inonda stderr di avvisi: qui si impone il trasporto corretto.
# Sul server aggiungere il binding:  --bind-to core --map-by core
MPIFLAGS ?= --mca btl self,sm

ifeq ($(PREC),float)
CFLAGS += -DUSE_FLOAT
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

OBJDIR := obj
OBJS   := $(patsubst src/%.c,$(OBJDIR)/%.o,$(SRCS))
DEPS   := $(OBJS:.o=.d)

BIN := bin/matmul_mpi
TESTBIN := bin/test_index

.PHONY: all test check clean

all: $(BIN) $(TESTBIN)

$(OBJDIR)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJS)
	@mkdir -p bin
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# I test degli indici non usano MPI: compilatore normale, binario autonomo.
$(TESTBIN): test/test_index.c src/index/index.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test: $(TESTBIN)
	./$(TESTBIN)

# Validazione: taglie volutamente NON divisibili per il numero di processi,
# cosi' i blocchi risultano disuguali ed eventuali off-by-one emergono.
# La stessa Y deve uscire da tutte le forme di griglia.
check: all
	./$(TESTBIN)
	mpirun -np 1 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 3  --pr 1 --pc 1 --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 6  --pr 1 --pc 4 --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 6  --pr 4 --pc 1 --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 8  --pr 2 --pc 2 --check --reps 2
	mpirun -np 3 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 32 --pr 3 --pc 1 --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 64  -N 512 -k 20 --pr 2 --pc 2 --check --reps 2

clean:
	rm -rf $(OBJDIR) bin

-include $(DEPS)
