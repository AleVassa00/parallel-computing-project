#ifndef SCPA_GEN_H
#define SCPA_GEN_H

#include <stdint.h>

#include "common/scalar.h"

/* Generazione riproducibile dei dati.
 *
 * Scelta progettuale centrale: il valore di ogni elemento e' una FUNZIONE PURA
 * degli indici GLOBALI, non lo stato di un generatore sequenziale.
 *
 * Conseguenze:
 *  - ogni processo genera direttamente il proprio blocco locale, senza che la
 *    matrice globale venga mai materializzata e senza alcuno scatter di A;
 *  - il contenuto di A e X non dipende dalla forma della griglia, quindi
 *    confrontare i risultati fra griglie diverse verifica gratuitamente la
 *    correttezza della distribuzione;
 *  - il seriale di riferimento puo' rigenerare A e X globali per la validazione.
 *
 * I valori stanno in [-1, 1) con media nulla: la somma su N termini cresce
 * come sqrt(N) e non produce cancellazioni catastrofiche che falserebbero
 * il confronto con il seriale. */

/* Stream distinti perche' A e X non abbiano lo stesso pattern di valori. */
#define GEN_STREAM_A 0x41ULL /* 'A' */
#define GEN_STREAM_X 0x58ULL /* 'X' */

#define GEN_DEFAULT_SEED 20252026ULL

scalar_t gen_value(uint64_t seed, uint64_t stream, uint64_t key);

/* Riempie il blocco locale di A: righe globali [row0, row0+m),
 * colonne globali [col0, col0+n). N e' il numero GLOBALE di colonne
 * ed entra nella chiave, quindi non e' ridondante con n. */
void gen_block_A(scalar_t *A, int lda, int m, int n,
                 int row0, int col0, int N, uint64_t seed);

/* Riempie il blocco locale di X: righe globali [row0, row0+n), k colonne.
 * k non viene mai partizionato, quindi non serve un col0. */
void gen_block_X(scalar_t *X, int ldx, int n, int k,
                 int row0, uint64_t seed);

#endif /* SCPA_GEN_H */
