/* Schema A: ordine dei cicli i -> j -> c, tutto row-major.
 *
 * E' lo schema che NON rilegge A k volte, ed e' il motivo per cui il prodotto
 * matrice-multivettore vale piu' di k prodotti matrice-vettore.
 *
 *   for i:
 *     acc[0..k) = 0
 *     for j:
 *       a = A[i][j]              <- letto UNA sola volta
 *       for c: acc[c] += a * X[j][c]   <- riusato k volte, dai registri
 *     Y[i][0..k) = acc[0..k)
 *
 * Proprieta':
 *  - entrambi gli stream in memoria sono a stride 1: la riga di A e la riga
 *    di X (che e' contigua perche' X e' row-major con k contiguo);
 *  - il riuso di A vale esattamente k, per qualunque k, senza casi degeneri.
 *    L'intensita' aritmetica passa da k/4 * (1/k) = 1/4 dello schema B a
 *    k/4 FLOP/byte, che e' il massimo ottenibile leggendo A una volta.
 *
 * Confronto con lo schema B (i -> c -> j, X column-major): li' la riga di A
 * viene riletta una volta per ogni colonna di X, e il riuso si recupera solo
 * con l'unrolling su c, che pero' satura a min(k, ampiezza del gruppo). Qui
 * il riuso e' k per costruzione.
 *
 * Vincolo: servono k accumulatori vivi contemporaneamente. Per k <= KB il
 * ciclo esterno su c0 gira una volta sola e A viene letta esattamente una
 * volta; per k > KB (fuori dal collaudo richiesto, ma il codice deve
 * funzionare per k generico) A viene riletta ceil(k/KB) volte, che resta il
 * comportamento migliore possibile a parita' di registri disponibili. */

#include "kernel/kernel.h"

#include <stddef.h>

/* Ampiezza del blocco di colonne tenuto negli accumulatori.
 * 32 copre tutti i k del collaudo (3, 6, 8, 20, 32): in quei casi il ciclo
 * su c0 e' degenere e A viene letta una volta sola. */
#define KB 32

void local_gemm(int m, int n, int k,
                const scalar_t *restrict A, int lda,
                const scalar_t *restrict X, int ldx,
                scalar_t *restrict Y, int ldy)
{
    int c0;

    for (c0 = 0; c0 < k; c0 += KB) {
        const int cw = (k - c0 < KB) ? (k - c0) : KB;
        int i;

        for (i = 0; i < m; i++) {
            const scalar_t *restrict arow = A + (size_t)i * (size_t)lda;
            scalar_t *restrict yrow = Y + (size_t)i * (size_t)ldy + c0;
            scalar_t acc[KB];
            int j, c;

            for (c = 0; c < cw; c++)
                acc[c] = (scalar_t)0;

            for (j = 0; j < n; j++) {
                const scalar_t a = arow[j];
                const scalar_t *restrict xrow = X + (size_t)j * (size_t)ldx + c0;
                /* Ciclo interno corto: con cw noto solo a runtime il
                 * compilatore genera versione vettoriale + coda scalare e
                 * tiene acc[] in stack (comunque residente in L1).
                 * La specializzazione sui k del collaudo, che permette di
                 * inchiodare acc[] nei registri vettoriali, e' il passo
                 * successivo sul kernel e non cambia questa interfaccia. */
                for (c = 0; c < cw; c++)
                    acc[c] += a * xrow[c];
            }

            for (c = 0; c < cw; c++)
                yrow[c] = acc[c];
        }
    }
}

const char *kernel_name(void)
{
    return "scheme_a";
}
