#ifndef SCPA_INDEX_H
#define SCPA_INDEX_H

/* Partizionamento a blocchi di n indici globali fra p parti.
 *
 * Regola unica: le prime (n % p) parti ricevono un elemento in piu'.
 * Le tre funzioni descrivono la STESSA partizione da tre angolazioni
 * (quanti, da dove, di chi) e devono restare coerenti fra loro.
 *
 * Questo modulo e' la sorgente piu' probabile di bug silenziosi:
 * indici sbagliati non fanno crashare nulla, producono solo numeri sbagliati.
 * Va testato isolatamente (test/test_index.c) prima di scrivere altro. */

/* Quanti elementi tocchino alla parte i. */
int block_size(int n, int p, int i);

/* Da quale indice globale comincia la parte i.
 * Vale anche per i == p: restituisce n (utile come sentinella). */
int block_start(int n, int p, int i);

/* Quale parte possiede l'indice globale g (0 <= g < n). */
int block_owner(int n, int p, int g);

#endif /* SCPA_INDEX_H */
