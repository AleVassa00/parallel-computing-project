#include "gen/gen.h"

#include <stddef.h>

/* splitmix64: mixer a 64 bit, stateless e di ottima qualita' statistica.
 * Serve proprio la sua natura stateless: da (seed, indici) al valore in un
 * passo solo, senza dover "avanzare" un generatore fino alla posizione. */
static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

scalar_t gen_value(uint64_t seed, uint64_t stream, uint64_t key)
{
    /* Il seme viene mescolato una volta con lo stream, poi con la chiave:
     * due passaggi evitano che chiavi vicine diano valori correlati. */
    uint64_t h = splitmix64(seed ^ (stream * 0x9E3779B97F4A7C15ULL));
    uint64_t r = splitmix64(h + key);
    /* 53 bit alti -> double esatto in [0,1), poi riscalato in [-1,1) */
    double u = (double)(r >> 11) * (1.0 / 9007199254740992.0);
    return (scalar_t)(2.0 * u - 1.0);
}

void gen_block_A(scalar_t *A, int lda, int m, int n,
                 int row0, int col0, int N, uint64_t seed)
{
    int i, j;
    for (i = 0; i < m; i++) {
        /* chiave = indice lineare globale row-major: i_glob * N + j_glob.
         * Il cast a 64 bit e' obbligatorio: a M = N = 40000 il prodotto
         * supera il range di int. */
        uint64_t base = (uint64_t)(row0 + i) * (uint64_t)N;
        scalar_t *row = A + (size_t)i * (size_t)lda;
        for (j = 0; j < n; j++)
            row[j] = gen_value(seed, GEN_STREAM_A, base + (uint64_t)(col0 + j));
    }
}

void gen_block_X(scalar_t *X, int ldx, int n, int k,
                 int row0, uint64_t seed)
{
    int j, c;
    for (j = 0; j < n; j++) {
        uint64_t base = (uint64_t)(row0 + j) * (uint64_t)k;
        scalar_t *row = X + (size_t)j * (size_t)ldx;
        for (c = 0; c < k; c++)
            row[c] = gen_value(seed, GEN_STREAM_X, base + (uint64_t)c);
    }
}
