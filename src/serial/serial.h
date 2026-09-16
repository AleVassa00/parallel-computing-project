#ifndef SERIAL_H
#define SERIAL_H

#include "common/scalar.h"

void serial_gemm(int m, int n, int k,
                 const scalar_t *A, int lda,
                 const scalar_t *X, int ldx,
                 scalar_t *Y, int ldy);

#endif
