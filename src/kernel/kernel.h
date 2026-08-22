#ifndef SCPA_KERNEL_H
#define SCPA_KERNEL_H

#include "common/scalar.h"

/* Interfaccia unica del kernel locale, indipendente dal backend.
 *
 * Il codice MPI non deve sapere quale implementazione gira sotto: schema A
 * scalare, OpenMP o CUDA. La scelta e' del Makefile (variabile KERNEL).
 *
 * Contratto:
 *   Y = A * X       (assegnazione, NON accumulo)
 *   A: m x n, riga i a partire da A + i*lda        row-major
 *   X: n x k, riga j a partire da X + j*ldx        row-major, k contiguo
 *   Y: m x k, riga i a partire da Y + i*ldy        row-major, k contiguo
 *
 * I tre buffer non devono sovrapporsi (sono dichiarati restrict).
 * Le leading dimension sono parametri e non coincidono necessariamente con
 * n e k: e' il gancio per il padding anti-conflict-miss (LDA = n + 24 e
 * simili) senza toccare ne' il kernel ne' il codice chiamante. */
void local_gemm(int m, int n, int k,
                const scalar_t *restrict A, int lda,
                const scalar_t *restrict X, int ldx,
                scalar_t *restrict Y, int ldy);

/* Nome del backend attivo, per l'intestazione delle misure. */
const char *kernel_name(void);

#endif /* SCPA_KERNEL_H */
