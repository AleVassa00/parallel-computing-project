#ifndef SCPA_CHECK_H
#define SCPA_CHECK_H

#include <stdint.h>

#include "common/scalar.h"
#include "mpi/distrib.h"
#include "mpi/grid.h"

/* Validazione del risultato distribuito contro il seriale.
 *
 * Il pezzo di Y che sta sulla colonna 0 della griglia viene raccolto sul
 * processo (0,0), che rigenera A e X GLOBALI dal seme e ricalcola Y in
 * seriale. La rigenerazione e' possibile solo perche' i valori sono funzione
 * degli indici globali: non c'e' bisogno di aver mai costruito la matrice
 * globale ne' di comunicare A.
 *
 * Il confronto e' l'errore relativo in norma L2, mai l'uguaglianza bit a bit:
 * la versione distribuita somma i contributi in un ordine diverso e
 * l'aritmetica IEEE 754 non e' associativa.
 *
 * Restituisce l'errore relativo, replicato su tutti i processi.
 * Costo: O(M*N) di memoria e O(M*N*k) di tempo sul rank 0, quindi va usato
 * su taglie modeste, non durante la campagna di misura. */
double check_against_serial(const grid_t *g, const layout_t *l,
                            const scalar_t *Y_loc, uint64_t seed);

#endif /* SCPA_CHECK_H */
