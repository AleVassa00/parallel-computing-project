#ifndef SCPA_SCALAR_H
#define SCPA_SCALAR_H

/* Tipo scalare unico del progetto.
 *
 * Il codice usa SOLO scalar_t: il confronto FP64 / FP32 diventa cosi' un flag
 * di compilazione (-DUSE_FLOAT) e non una seconda copia dei sorgenti.
 *
 * SCALAR_MPI_TYPE e' una macro: si espande solo dove viene usata, quindi
 * questo header resta includibile anche da unita' che non vedono mpi.h
 * (per esempio i test delle funzioni indice). */

#ifdef USE_FLOAT
typedef float scalar_t;
#define SCALAR_MPI_TYPE  MPI_FLOAT
#define SCALAR_NAME      "float"
#define SCALAR_CHECK_TOL 1.0e-5    /* ~2^-23 amplificato dalla riduzione su N */
#else
typedef double scalar_t;
#define SCALAR_MPI_TYPE  MPI_DOUBLE
#define SCALAR_NAME      "double"
#define SCALAR_CHECK_TOL 1.0e-12
#endif

#endif /* SCPA_SCALAR_H */
