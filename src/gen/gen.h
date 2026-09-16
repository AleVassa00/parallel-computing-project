#ifndef GEN_H
#define GEN_H

#include <stdint.h>

#include "common/scalar.h"

#define GEN_STREAM_A 0x41ULL
#define GEN_STREAM_X 0x58ULL

#define GEN_DEFAULT_SEED 20252026ULL

scalar_t gen_value(uint64_t seed, uint64_t stream, uint64_t key);

void gen_block_A(scalar_t *A, int lda, int m, int n,
                 int row0, int col0, int N, uint64_t seed);

void gen_block_X(scalar_t *X, int ldx, int n, int k,
                 int row0, uint64_t seed);

#endif
