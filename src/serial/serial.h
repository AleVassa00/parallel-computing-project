#ifndef SCPA_SERIAL_H
#define SCPA_SERIAL_H

#include "common/scalar.h"

/* Implementazione seriale di riferimento: Y = A*X, tutto row-major.
 *
 * Non e' un kernel da misurare, e' l'ORACOLO della validazione. Per questo e'
 * scritta nel modo piu' diretto possibile a partire dalla definizione
 * (ordine i-c-j) e deliberatamente con un ordine di cicli DIVERSO da quello
 * dello schema A: se il ragionamento sull'ordine dei cicli fosse sbagliato,
 * un oracolo con la stessa struttura nasconderebbe l'errore invece di
 * rivelarlo. */
void serial_gemm(int m, int n, int k,
                 const scalar_t *A, int lda,
                 const scalar_t *X, int ldx,
                 scalar_t *Y, int ldy);

#endif /* SCPA_SERIAL_H */
