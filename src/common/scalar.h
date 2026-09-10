#ifndef SCALAR_H
#define SCALAR_H

#include <float.h>
#include <math.h>

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
#define SCALAR_EPS       ((double)FLT_EPSILON)
#else
typedef double scalar_t;
#define SCALAR_MPI_TYPE  MPI_DOUBLE
#define SCALAR_NAME      "double"
#define SCALAR_EPS       ((double)DBL_EPSILON)
#endif

/* ---------------------------------------------------------------------------
 * Tolleranza della validazione
 * ---------------------------------------------------------------------------
 * NON e' una costante, e non puo' esserlo: ogni elemento di Y e' una riduzione
 * su n termini, e l'errore di arrotondamento di una riduzione dipende da n.
 *
 * Con dati a media nulla - che e' esattamente cio' che genera src/gen, e per
 * questo motivo - gli arrotondamenti hanno segno indipendente e l'errore
 * cresce come eps*sqrt(n), non come il pessimistico eps*n del caso in cui
 * tutti i termini abbiano lo stesso segno.
 *
 * La costante del modello e' MISURATA, non stimata: confrontando il prodotto
 * distribuito con l'oracolo seriale su n = 257 ... 40000, in entrambe le
 * precisioni, l'errore relativo L2 vale
 *
 *     err  ~=  0.17 * eps * sqrt(n)
 *
 * con un rapporto che resta fra 0.11 e 0.17 su tutto l'intervallo: il modello
 * sqrt e' quello giusto, e la soglia puo' quindi essere stretta.
 *
 * Una soglia FISSA sbaglia invece in due modi opposti, e li sbaglia entrambi
 * nell'intervallo di taglie che il progetto misura davvero:
 *
 *   double, 1e-12  ->  TROPPO LARGA. A n = 257 l'errore reale e' 5.6e-16,
 *                      cioe' la soglia era ~1800 volte l'errore; a n = 40000
 *                      ancora ~140 volte. Un bug che avesse gonfiato l'errore
 *                      di due ordini di grandezza sarebbe passato per PASS.
 *   float,  1e-5   ->  SI RESTRINGE al crescere di n. Il margine passa da 33x
 *                      a n = 257 a 2.5x a n = 40000, che e' la taglia a cui
 *                      la campagna lavora. 2.5x non e' un margine: basta una
 *                      distribuzione dei dati un po' diversa, o un backend con
 *                      un altro ordine di riduzione, per farlo diventare un
 *                      FAIL su un risultato corretto - e siccome il binario
 *                      esce con stato non nullo, quel FAIL abortisce la
 *                      campagna di misura.
 *
 * n e' la lunghezza della riduzione, cioe' il numero GLOBALE di colonne di A
 * (N), non n_loc: il riferimento seriale somma su tutte e N.
 *
 * SCALAR_CHECK_SAFETY e' il margine sul modello. Con la costante misurata di
 * 0.17, il valore 8 lascia un fattore ~47x uniforme su tutte le taglie e su
 * entrambe le precisioni: abbastanza per non dare falsi FAIL cambiando dati,
 * griglia o backend, e abbastanza stretto da restare un controllo vero - un
 * indice sbagliato produce un errore relativo di ordine 1, non 47 volte
 * eps*sqrt(n). */
#define SCALAR_CHECK_SAFETY 8.0

static inline double scalar_check_tol(int n)
{
    const double terms = (n > 1) ? (double)n : 1.0;
    return SCALAR_CHECK_SAFETY * SCALAR_EPS * sqrt(terms);
}

#endif /* SCALAR_H */
