
CC     := gcc
CXX    := g++
MPICC  := mpicc
NVCC   ?= nvcc

NVCC_ARCH ?= sm_75

KERNEL ?= scheme_a
PREC   ?= double
FORCE_GENERIC_K ?= 0
TEST_A_PADDING  ?= 0

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

SMEM_PAD        ?= 1

TILE_GRANULARITY ?= 32

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

BLOCK           ?= 256
EXTRA_CFLAGS    ?=
EXTRA_NVCCFLAGS ?=

ARCHFLAGS := $(shell \
    $(CC) -march=native -E -x c /dev/null >/dev/null 2>&1 && echo -march=native || \
    ($(CC) -mcpu=native -E -x c /dev/null >/dev/null 2>&1 && echo -mcpu=native))

CFLAGS := -std=c11 -O3 $(ARCHFLAGS) -ffp-contract=fast \
	-Wall -Wextra -Wpedantic -Isrc -MMD -MP \
	$(EXTRA_CFLAGS)

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
NVCCFLAGS := -O3 -std=c++14 -arch=$(NVCC_ARCH) -Isrc $(PRECDEF) -lineinfo \
	$(X_LAYOUT_DEF) \
	-DSMEM_PAD=$(SMEM_PAD) -DBLOCK_THREADS=$(BLOCK) \
	-Xptxas -v \
	-Xcompiler -Wall -Xcompiler -Wextra $(EXTRA_NVCCFLAGS)
ifeq ($(KERNEL),cuda_warp_smem)
NVCCFLAGS += -DTILE_GRANULARITY=$(TILE_GRANULARITY)
ifneq ($(SMEM_BUDGET_BYTES),)
NVCCFLAGS += -DSMEM_BUDGET_BYTES=$(SMEM_BUDGET_BYTES)
endif
endif
ifneq ($(ARCHFLAGS),)
NVCCFLAGS += -Xcompiler $(ARCHFLAGS)
endif

ifeq ($(origin CUDA_HOME), undefined)
CUDA_HOME := $(patsubst %/bin/,%,$(dir $(shell command -v $(NVCC) 2>/dev/null)))
endif
ifneq ($(CUDA_HOME),)
LDFLAGS += -L$(CUDA_HOME)/lib64
endif
LDLIBS += -lcudart -lstdc++

ifneq ($(shell grep -l cublas_v2.h $(KERNEL_SRC) 2>/dev/null),)
LDLIBS += -lcublas
endif
endif

OBJDIR ?= obj/matmul_mpi$(CONFIG)
KERNEL_OBJ := $(OBJDIR)/kernel/$(KERNEL).o
OBJS   := $(patsubst src/%.c,$(OBJDIR)/%.o,$(C_SRCS)) $(KERNEL_OBJ)
DEPS   := $(patsubst src/%.c,$(OBJDIR)/%.d,$(C_SRCS))

BIN ?= bin/matmul_mpi$(CONFIG)
TESTBIN := bin/test_index

.PHONY: all test check check-mpi check-cxx check-padding padding-run clean

all: $(BIN)
	@echo "built $(BIN)  [PREC=$(PREC) KERNEL=$(KERNEL) ($(KERNEL_SRC)) FORCE_GENERIC_K=$(FORCE_GENERIC_K) TEST_A_PADDING=$(TEST_A_PADDING) SMEM_PAD=$(SMEM_PAD) BLOCK=$(BLOCK) TILE_GRANULARITY=$(TILE_GRANULARITY) SMEM_BUDGET_BYTES=$(if $(SMEM_BUDGET_BYTES),$(SMEM_BUDGET_BYTES),derivato)$(if $(filter cuda_warp,$(KERNEL)), X_LAYOUT=$(X_LAYOUT))]"

$(OBJDIR)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -c $< -o $@

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

$(BIN): $(OBJS)
	@mkdir -p bin
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(TESTBIN): test/test_index.c src/index/index.c
	@mkdir -p bin
	$(CC) $(CFLAGS) $^ -o $@ -lm

test: $(TESTBIN)
	./$(TESTBIN)

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
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 2 -N 3 -k 7 --pr 4 --pc 1 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 2 -N 3 -k 7 --pr 4 --pc 1 --a-mode global --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 3 -N 2 -k 7 --pr 1 --pc 4 --a-mode local  --check --reps 2
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 3 -N 2 -k 7 --pr 1 --pc 4 --a-mode global --x-mode global --check --reps 2

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

check-padding:
	$(MAKE) TEST_A_PADDING=8 padding-run

padding-run: $(BIN)
	mpirun -np 4 $(MPIFLAGS) ./$(BIN) -M 301 -N 173 -k 8 \
		--pr 2 --pc 2 --a-mode global --check --reps 2

clean:
	rm -rf obj bin

-include $(DEPS)
