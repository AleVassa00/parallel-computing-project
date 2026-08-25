#ifndef SCPA_MATMUL_MPI_H
#define SCPA_MATMUL_MPI_H

#include "common/scalar.h"
#include "kernel/kernel.h"
#include "mpi/distrib.h"
#include "mpi/grid.h"

/* Tempi della singola invocazione, misurati localmente da ciascun processo.
 * Il tempo di fase include l'attesa dei processi in ritardo: e' voluto, e'
 * esattamente il costo che si paga sul cammino critico. */
typedef struct {
    double t_bcast;
    double t_local;
    double t_reduce;
    double t_total;
} matmul_time_t;

/* Y = A*X distribuito. Tutti i processi eseguono lo stesso codice.
 *
 *   1. MPI_Bcast di X_loc lungo col_comm            (root = riga 0)
 *   2. Y_parz = A_loc * X_loc                      (kernel locale)
 *   3. MPI_Reduce(MPI_SUM) lungo row_comm          (root = colonna 0)
 *
 * All'ingresso X_loc e' significativo soltanto sulla grid row 0. Il broadcast
 * lo replica nella relativa colonna; la reduce ricompone Y perche' ogni
 * processo di una riga ha calcolato soltanto il contributo delle colonne di A
 * che possiede.
 *
 * kern:  contesto del kernel locale, creato in preprocessing con il blocco
 *        A_loc di questo processo. A non compare piu' qui: il backend la
 *        possiede gia' (su GPU e' la copia in VRAM), e forma e leading
 *        dimension vengono da li', non possono divergere dal layout.
 * X_loc: fetta n_loc x k inizializzata sulla grid row 0; buffer su tutti.
 * Ypart: buffer di lavoro m_loc x k, richiesto su tutti i processi.
 * Y_loc: significativo solo sulla colonna 0; puo' essere NULL altrove.
 *        Volutamente distinto da Ypart, cosi' non serve MPI_IN_PLACE.
 * t:     puo' essere NULL se non interessa la scomposizione dei tempi. */
void mpi_matmul(const grid_t *g, const layout_t *l,
                local_gemm_t *kern,
                scalar_t *X_loc,
                scalar_t *Ypart,
                scalar_t *Y_loc,
                matmul_time_t *t);

#endif /* SCPA_MATMUL_MPI_H */
