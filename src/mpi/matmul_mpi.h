#ifndef SCPA_MATMUL_MPI_H
#define SCPA_MATMUL_MPI_H

#include "common/scalar.h"
#include "mpi/distrib.h"
#include "mpi/grid.h"

/* Tempi della singola invocazione, misurati localmente da ciascun processo.
 * Il tempo di fase include l'attesa dei processi in ritardo: e' voluto, e'
 * esattamente il costo che si paga sul cammino critico. */
typedef struct {
    double t_local;
    double t_reduce;
    double t_total;
} matmul_time_t;

/* Replica una sola volta la fetta di X generata sulla grid row 0 lungo il
 * column communicator. Restituisce il tempo locale del broadcast, utile come
 * metrica di setup separata dal benchmark. */
double mpi_distribute_X(const grid_t *g, const layout_t *l, scalar_t *X_loc);

/* Y = A*X distribuito. Tutti i processi eseguono lo stesso codice.
 *
 *   1. Y_parz = A_loc * X_loc                      (kernel locale)
 *   2. MPI_Reduce(MPI_SUM) lungo row_comm          (root = colonna 0 della griglia)
 *
 * Prima della chiamata X_loc deve essere gia' disponibile su tutti i processi
 * della stessa colonna, normalmente tramite mpi_distribute_X(). La reduce
 * ricompone Y perche' ogni processo di una riga ha calcolato soltanto il
 * contributo delle colonne di A che possiede.
 *
 * X_loc: fetta n_loc x k gia' replicata su tutta la colonna.
 * Ypart: buffer di lavoro m_loc x k, richiesto su tutti i processi.
 * Y_loc: significativo solo sulla colonna 0; puo' essere NULL altrove.
 *        Volutamente distinto da Ypart, cosi' non serve MPI_IN_PLACE.
 * t:     puo' essere NULL se non interessa la scomposizione dei tempi. */
void mpi_matmul(const grid_t *g, const layout_t *l,
                const scalar_t *A_loc,
                const scalar_t *X_loc,
                scalar_t *Ypart,
                scalar_t *Y_loc,
                matmul_time_t *t);

#endif /* SCPA_MATMUL_MPI_H */
