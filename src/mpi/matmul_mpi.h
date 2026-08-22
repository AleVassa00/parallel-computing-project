#ifndef SCPA_MATMUL_MPI_H
#define SCPA_MATMUL_MPI_H

#include "common/scalar.h"
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
 *   1. MPI_Bcast della fetta di X lungo col_comm   (root = riga 0 della griglia)
 *   2. Y_parz = A_loc * X_loc                      (kernel locale)
 *   3. MPI_Reduce(MPI_SUM) lungo row_comm          (root = colonna 0 della griglia)
 *
 * Il broadcast e' l'unico modo sensato di soddisfare tutti i processi di una
 * colonna, che vogliono la stessa fetta di X; la reduce e' l'unico modo di
 * ricomporre Y, perche' ogni processo di una riga ha calcolato solo il
 * contributo delle colonne di A che possiede.
 *
 * X_loc: in ingresso significativo solo sulla riga 0 della griglia, in uscita
 *        replicato su tutta la colonna.
 * Ypart: buffer di lavoro m_loc x k, richiesto su tutti i processi.
 * Y_loc: significativo solo sulla colonna 0; puo' essere NULL altrove.
 *        Volutamente distinto da Ypart, cosi' non serve MPI_IN_PLACE.
 * t:     puo' essere NULL se non interessa la scomposizione dei tempi. */
void mpi_matmul(const grid_t *g, const layout_t *l,
                const scalar_t *A_loc,
                scalar_t *X_loc,
                scalar_t *Ypart,
                scalar_t *Y_loc,
                matmul_time_t *t);

#endif /* SCPA_MATMUL_MPI_H */
