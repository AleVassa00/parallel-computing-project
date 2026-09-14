/* Test isolato delle funzioni indice.
 *
 * Non usa MPI: verifica solo l'aritmetica della partizione, che e' il
 * presupposto di tutto il resto (distribuzione di A, X e Y).
 *
 * Proprieta' verificate, per ogni coppia (n, p):
 *   1. le parti coprono esattamente [0, n) senza sovrapposizioni;
 *   2. block_start e block_size sono coerenti fra loro;
 *   3. lo sbilanciamento e' al piu' 1 elemento, e le parti lunghe vengono prima;
 *   4. block_owner e' l'inversa di block_start/block_size.
 * Piu' i casi espliciti citati nel documento di contesto. */

#include "index/index.h"

#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__);                      \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void test_partition(int n, int p)
{
    int i, g, total = 0, prev_size = -1;

    for (i = 0; i < p; i++) {
        int s = block_size(n, p, i);
        int b = block_start(n, p, i);

        CHECK(s >= 0, "n=%d p=%d i=%d: negative size %d", n, p, i, s);
        CHECK(b == total, "n=%d p=%d i=%d: start %d, expected %d", n, p, i, b, total);

        /* le parti lunghe (q+1) precedono quelle corte (q) */
        if (prev_size >= 0)
            CHECK(s <= prev_size && s >= prev_size - 1,
                  "n=%d p=%d i=%d: sizes not monotone (%d after %d)", n, p, i, s, prev_size);
        prev_size = s;
        total += s;
    }

    CHECK(total == n, "n=%d p=%d: parts cover %d elements, expected %d", n, p, total, n);
    CHECK(block_start(n, p, p) == n, "n=%d p=%d: sentinel start(p) = %d, expected %d",
          n, p, block_start(n, p, p), n);

    /* block_owner deve riportare ogni indice globale nella sua parte */
    for (g = 0; g < n; g++) {
        int o = block_owner(n, p, g);
        int b = block_start(n, p, o);
        int s = block_size(n, p, o);
        CHECK(o >= 0 && o < p, "n=%d p=%d g=%d: owner %d out of range", n, p, g, o);
        CHECK(g >= b && g < b + s,
              "n=%d p=%d g=%d: owner %d covers [%d,%d)", n, p, g, o, b, b + s);
    }
}

int main(void)
{
    int n, p;

    printf("test_index: block_size / block_start / block_owner\n");

    /* casi espliciti del documento di contesto */
    CHECK(block_size(10, 3, 0) == 4, "block_size(10,3,0) = %d", block_size(10, 3, 0));
    CHECK(block_size(10, 3, 1) == 3, "block_size(10,3,1) = %d", block_size(10, 3, 1));
    CHECK(block_size(10, 3, 2) == 3, "block_size(10,3,2) = %d", block_size(10, 3, 2));
    CHECK(block_start(10, 3, 0) == 0, "block_start(10,3,0) = %d", block_start(10, 3, 0));
    CHECK(block_start(10, 3, 1) == 4, "block_start(10,3,1) = %d", block_start(10, 3, 1));
    CHECK(block_start(10, 3, 2) == 7, "block_start(10,3,2) = %d", block_start(10, 3, 2));

    /* caso divisibile */
    CHECK(block_size(12, 4, 3) == 3, "block_size(12,4,3) = %d", block_size(12, 4, 3));
    CHECK(block_start(12, 4, 3) == 9, "block_start(12,4,3) = %d", block_start(12, 4, 3));

    /* piu' parti che elementi: le ultime restano vuote */
    CHECK(block_size(3, 5, 2) == 1, "block_size(3,5,2) = %d", block_size(3, 5, 2));
    CHECK(block_size(3, 5, 3) == 0, "block_size(3,5,3) = %d", block_size(3, 5, 3));
    CHECK(block_start(3, 5, 4) == 3, "block_start(3,5,4) = %d", block_start(3, 5, 4));

    /* sweep esaustivo sui casi piccoli, dove vivono gli off-by-one */
    for (n = 0; n <= 64; n++)
        for (p = 1; p <= 16; p++)
            test_partition(n, p);

    /* qualche caso realistico, incluse le forme M = 3N e N = 2M */
    {
        const int sizes[] = { 1000, 4096, 10000, 12000, 30000 };
        const int parts[] = { 1, 2, 3, 4, 5, 7, 8, 10, 16, 20 };
        size_t a, b;
        for (a = 0; a < sizeof sizes / sizeof *sizes; a++)
            for (b = 0; b < sizeof parts / sizeof *parts; b++)
                test_partition(sizes[a], parts[b]);
    }

    if (failures == 0) {
        printf("test_index: all checks passed\n");
        return EXIT_SUCCESS;
    }
    printf("test_index: %d failure(s)\n", failures);
    return EXIT_FAILURE;
}
