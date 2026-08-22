#include "serial/serial.h"

#include <stddef.h>

void serial_gemm(int m, int n, int k,
                 const scalar_t *A, int lda,
                 const scalar_t *X, int ldx,
                 scalar_t *Y, int ldy)
{
    int i, j, c;
    for (i = 0; i < m; i++) {
        for (c = 0; c < k; c++) {
            scalar_t s = (scalar_t)0;
            for (j = 0; j < n; j++)
                s += A[(size_t)i * lda + j] * X[(size_t)j * ldx + c];
            Y[(size_t)i * ldy + c] = s;
        }
    }
}
