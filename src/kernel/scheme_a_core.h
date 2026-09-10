#ifndef SCPA_KERNEL_SCHEME_A_CORE_H
#define SCPA_KERNEL_SCHEME_A_CORE_H

/* Nucleo scalare dello schema A, condiviso da tutti i backend di CPU.
 *
 * Lo schema e' quello descritto in src/kernel/scheme_a.c: ordine dei cicli
 * i -> j -> c, A letta una sola volta, k accumulatori tenuti in registro.
 *
 *   for i:
 *     acc[0..k) = 0                  (oppure = Y[i][0..k), vedi sotto)
 *     for j:
 *       a = A[i][j]                  <- letto UNA sola volta
 *       for c: acc[c] += a * X[j][c] <- riusato k volte, dai registri
 *     Y[i][0..k) = acc[0..k)
 *
 * Qui il nucleo vive in un header, e non dentro scheme_a.c, perche' i backend
 * di CPU sono ormai tre - scalare, OpenMP e OpenMP con tiling di cache - e
 * differiscono SOLO per come distribuiscono il lavoro, non per come lo fanno.
 * Tenere una copia del ciclo caldo per backend significherebbe misurare tre
 * kernel diversi credendo di misurare tre parallelizzazioni della stessa cosa:
 * il confronto fra i backend sarebbe un aneddoto, non un esperimento
 * controllato. Con il nucleo condiviso l'unica variabile che cambia e' la
 * mappatura sui thread, che e' esattamente cio' che si vuole misurare.
 *
 * ---------------------------------------------------------------------------
 * Due sole differenze rispetto alla versione che stava in scheme_a.c
 * ---------------------------------------------------------------------------
 *  1. i cicli sono su INTERVALLI [i0,i1) x [j0,j1) invece che su 0..m e 0..n.
 *     L'intervallo di righe serve a OpenMP (ogni thread possiede una fetta di
 *     righe di Y, disgiunta da quelle degli altri); l'intervallo di colonne
 *     serve al tiling di cache (una fetta di X alla volta).
 *  2. esistono due famiglie: _assign scrive Y = A*X sull'intervallo, _accum
 *     scrive Y += A*X. La seconda serve solo a chi spezza il ciclo su j: se
 *     l'intervallo delle colonne e' parziale, il contributo dei tile
 *     successivi va SOMMATO a quello dei precedenti. Sono due famiglie
 *     distinte, generate dalle stesse liste di colonne, e non un unico kernel
 *     con un flag: un flag runtime metterebbe un ramo per riga nel percorso
 *     caldo di TUTTI i backend, incluso quello scalare che non ne ha bisogno.
 *
 * Il contratto di scheme_a resta identico: chi chiama con [0,m) x [0,n) e la
 * famiglia _assign ottiene esattamente il codice di prima.
 *
 * FORCE_GENERIC_K vale per tutti i backend che includono questo header: e' un
 * -D del Makefile e disattiva la specializzazione su k, cosi' il microbenchmark
 * "k noto a compile-time contro fallback generico" si puo' rifare su ogni
 * backend, non solo su quello scalare. */

#include <stddef.h>

#include "kernel/kernel.h"

/* Ampiezza del blocco di colonne tenuto negli accumulatori dal kernel
 * generico. 32 copre tutti i k del collaudo (3, 6, 8, 20, 32): in quei casi il
 * ciclo su c0 e' degenere e A viene letta una volta sola. */
#define SCHEME_A_KB 32

/* Un header di nuclei intercambiabili definisce piu' funzioni di quante ogni
 * singolo backend ne usi: quello scalare non tocca mai la famiglia _accum.
 * Senza questo attributo -Wunused-function segnalerebbe come errore di
 * scrittura cio' che e' invece la natura del file. Non si usa `static inline`
 * proprio per non cambiare nulla rispetto al codice che stava in scheme_a.c:
 * le funzioni restano `static`, e la decisione se espanderle resta quella che
 * il compilatore prendeva prima. */
#if defined(__GNUC__)
#define SCHEME_A_UNUSED __attribute__((unused))
#else
#define SCHEME_A_UNUSED
#endif

/* ---------------------------------------------------------------------------
 * Fallback generico su k
 * --------------------------------------------------------------------------- */

#define SCHEME_A_DEFINE_GENERIC(NAME, ACC_INIT)                                \
    static SCHEME_A_UNUSED void NAME(int i0, int i1, int j0, int j1, int k,    \
                                     const scalar_t *RESTRICT A_loc, int lda,  \
                                     const scalar_t *RESTRICT X_loc, int ldx,  \
                                     scalar_t *RESTRICT Y_loc_part, int ldy)   \
    {                                                                          \
        int c0;                                                                \
                                                                               \
        for (c0 = 0; c0 < k; c0 += SCHEME_A_KB) {                              \
            const int cw = (k - c0 < SCHEME_A_KB) ? (k - c0) : SCHEME_A_KB;    \
            int i;                                                             \
                                                                               \
            for (i = i0; i < i1; i++) {                                        \
                const scalar_t *RESTRICT arow =                                \
                    A_loc + (size_t)i * (size_t)lda;                           \
                scalar_t *RESTRICT yrow =                                      \
                    Y_loc_part + (size_t)i * (size_t)ldy + c0;                 \
                scalar_t acc[SCHEME_A_KB];                                     \
                int j, c;                                                      \
                                                                               \
                for (c = 0; c < cw; c++)                                       \
                    acc[c] = ACC_INIT;                                         \
                                                                               \
                for (j = j0; j < j1; j++) {                                    \
                    const scalar_t a = arow[j];                                \
                    const scalar_t *RESTRICT xrow =                            \
                        X_loc + (size_t)j * (size_t)ldx + c0;                  \
                    for (c = 0; c < cw; c++)                                   \
                        acc[c] += a * xrow[c];                                 \
                }                                                              \
                                                                               \
                for (c = 0; c < cw; c++)                                       \
                    yrow[c] = acc[c];                                          \
            }                                                                  \
        }                                                                      \
    }

SCHEME_A_DEFINE_GENERIC(scheme_a_generic_assign, (scalar_t)0)
SCHEME_A_DEFINE_GENERIC(scheme_a_generic_accum,  yrow[c])

/* ---------------------------------------------------------------------------
 * Specializzazioni sui k richiesti dalla traccia
 * ---------------------------------------------------------------------------
 * Una sola lista per ogni ampiezza genera dichiarazione, aggiornamento e
 * store. Il percorso caldo risultante contiene istruzioni C esplicite per
 * ogni colonna, senza un limite runtime sul ciclo c. */

#ifndef FORCE_GENERIC_K

#define SCHEME_A_COLS_3(M)  M(0) M(1) M(2)
#define SCHEME_A_COLS_6(M)  SCHEME_A_COLS_3(M) M(3) M(4) M(5)
#define SCHEME_A_COLS_8(M)  SCHEME_A_COLS_6(M) M(6) M(7)
#define SCHEME_A_COLS_20(M) SCHEME_A_COLS_8(M) M(8) M(9) M(10) M(11) M(12) \
                            M(13) M(14) M(15) M(16) M(17) M(18) M(19)
#define SCHEME_A_COLS_32(M) SCHEME_A_COLS_20(M) M(20) M(21) M(22) M(23) \
                            M(24) M(25) M(26) M(27) M(28) M(29) M(30) M(31)

#define SCHEME_A_DECL_ZERO(c) scalar_t acc##c = (scalar_t)0;
#define SCHEME_A_DECL_LOAD(c) scalar_t acc##c = yrow[c];
#define SCHEME_A_UPDATE(c)    acc##c += a * xrow[c];
#define SCHEME_A_STORE(c)     yrow[c] = acc##c;

#define SCHEME_A_DEFINE_FIXED(NAME, COLS, DECL)                                \
    static SCHEME_A_UNUSED void NAME(int i0, int i1, int j0, int j1,           \
                                     const scalar_t *RESTRICT A_loc, int lda,  \
                                     const scalar_t *RESTRICT X_loc, int ldx,  \
                                     scalar_t *RESTRICT Y_loc_part, int ldy)   \
    {                                                                          \
        int i;                                                                 \
        for (i = i0; i < i1; i++) {                                            \
            const scalar_t *RESTRICT arow =                                    \
                A_loc + (size_t)i * (size_t)lda;                               \
            scalar_t *RESTRICT yrow =                                          \
                Y_loc_part + (size_t)i * (size_t)ldy;                          \
            int j;                                                             \
            COLS(DECL)                                                         \
            for (j = j0; j < j1; j++) {                                        \
                const scalar_t a = arow[j];                                    \
                const scalar_t *RESTRICT xrow =                                \
                    X_loc + (size_t)j * (size_t)ldx;                           \
                COLS(SCHEME_A_UPDATE)                                          \
            }                                                                  \
            COLS(SCHEME_A_STORE)                                               \
        }                                                                      \
    }

SCHEME_A_DEFINE_FIXED(scheme_a_k3_assign,  SCHEME_A_COLS_3,  SCHEME_A_DECL_ZERO)
SCHEME_A_DEFINE_FIXED(scheme_a_k6_assign,  SCHEME_A_COLS_6,  SCHEME_A_DECL_ZERO)
SCHEME_A_DEFINE_FIXED(scheme_a_k8_assign,  SCHEME_A_COLS_8,  SCHEME_A_DECL_ZERO)
SCHEME_A_DEFINE_FIXED(scheme_a_k20_assign, SCHEME_A_COLS_20, SCHEME_A_DECL_ZERO)
SCHEME_A_DEFINE_FIXED(scheme_a_k32_assign, SCHEME_A_COLS_32, SCHEME_A_DECL_ZERO)

SCHEME_A_DEFINE_FIXED(scheme_a_k3_accum,   SCHEME_A_COLS_3,  SCHEME_A_DECL_LOAD)
SCHEME_A_DEFINE_FIXED(scheme_a_k6_accum,   SCHEME_A_COLS_6,  SCHEME_A_DECL_LOAD)
SCHEME_A_DEFINE_FIXED(scheme_a_k8_accum,   SCHEME_A_COLS_8,  SCHEME_A_DECL_LOAD)
SCHEME_A_DEFINE_FIXED(scheme_a_k20_accum,  SCHEME_A_COLS_20, SCHEME_A_DECL_LOAD)
SCHEME_A_DEFINE_FIXED(scheme_a_k32_accum,  SCHEME_A_COLS_32, SCHEME_A_DECL_LOAD)

#endif /* FORCE_GENERIC_K */

/* ---------------------------------------------------------------------------
 * Dispatch su k
 * ---------------------------------------------------------------------------
 * E' l'unico punto in cui i cinque k obbligatori vengono nominati, e vale per
 * tutti i backend di CPU. Un k qualsiasi cade nel fallback generico: il codice
 * "dovra' poter funzionare per k generico" anche quando la specializzazione
 * non c'e'. */

/* Y[i0:i1, :] = A[i0:i1, j0:j1] * X[j0:j1, :] */
static SCHEME_A_UNUSED void scheme_a_assign(int i0, int i1, int j0, int j1, int k,
                                            const scalar_t *RESTRICT A_loc, int lda,
                                            const scalar_t *RESTRICT X_loc, int ldx,
                                            scalar_t *RESTRICT Y_loc_part, int ldy)
{
#ifdef FORCE_GENERIC_K
    scheme_a_generic_assign(i0, i1, j0, j1, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
#else
    switch (k) {
    case 3:  scheme_a_k3_assign (i0, i1, j0, j1, A_loc, lda, X_loc, ldx, Y_loc_part, ldy); break;
    case 6:  scheme_a_k6_assign (i0, i1, j0, j1, A_loc, lda, X_loc, ldx, Y_loc_part, ldy); break;
    case 8:  scheme_a_k8_assign (i0, i1, j0, j1, A_loc, lda, X_loc, ldx, Y_loc_part, ldy); break;
    case 20: scheme_a_k20_assign(i0, i1, j0, j1, A_loc, lda, X_loc, ldx, Y_loc_part, ldy); break;
    case 32: scheme_a_k32_assign(i0, i1, j0, j1, A_loc, lda, X_loc, ldx, Y_loc_part, ldy); break;
    default:
        scheme_a_generic_assign(i0, i1, j0, j1, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    }
#endif
}

/* Y[i0:i1, :] += A[i0:i1, j0:j1] * X[j0:j1, :] */
static SCHEME_A_UNUSED void scheme_a_accum(int i0, int i1, int j0, int j1, int k,
                                           const scalar_t *RESTRICT A_loc, int lda,
                                           const scalar_t *RESTRICT X_loc, int ldx,
                                           scalar_t *RESTRICT Y_loc_part, int ldy)
{
#ifdef FORCE_GENERIC_K
    scheme_a_generic_accum(i0, i1, j0, j1, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
#else
    switch (k) {
    case 3:  scheme_a_k3_accum (i0, i1, j0, j1, A_loc, lda, X_loc, ldx, Y_loc_part, ldy); break;
    case 6:  scheme_a_k6_accum (i0, i1, j0, j1, A_loc, lda, X_loc, ldx, Y_loc_part, ldy); break;
    case 8:  scheme_a_k8_accum (i0, i1, j0, j1, A_loc, lda, X_loc, ldx, Y_loc_part, ldy); break;
    case 20: scheme_a_k20_accum(i0, i1, j0, j1, A_loc, lda, X_loc, ldx, Y_loc_part, ldy); break;
    case 32: scheme_a_k32_accum(i0, i1, j0, j1, A_loc, lda, X_loc, ldx, Y_loc_part, ldy); break;
    default:
        scheme_a_generic_accum(i0, i1, j0, j1, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    }
#endif
}

/* ---------------------------------------------------------------------------
 * Partizione statica delle righe fra i thread
 * ---------------------------------------------------------------------------
 * Il costo di una riga di Y e' n_loc moltiplicazioni-addizioni per OGNI riga,
 * senza eccezioni: il lavoro e' perfettamente bilanciato per costruzione e non
 * c'e' niente da correggere a runtime. Percio' la partizione e' STATICA e a
 * blocchi contigui, che e' anche l'unica forma che conserva le due proprieta'
 * che rendono veloce il nucleo:
 *
 *   - ogni thread percorre una fetta CONTIGUA di A, quindi il prefetcher
 *     hardware vede uno stream lineare per thread invece di k stream
 *     intrecciati come farebbe una schedule ciclica;
 *   - le righe di Y scritte da thread diversi sono disgiunte e contigue,
 *     quindi il false sharing si limita a UNA cache line per confine fra
 *     thread, e solo quando k*sizeof(scalar_t) < 64 byte (cioe' k < 8 in
 *     doppia precisione). Una schedule ciclica renderebbe invece condivisa
 *     ogni linea di Y.
 *
 * Il resto m % nt viene distribuito una riga a testa sui primi thread: la
 * differenza di carico fra due thread e' cosi' al piu' una riga. */
static SCHEME_A_UNUSED void scheme_a_row_range(int m, int nthreads, int tid,
                                               int *i0, int *i1)
{
    const int base = m / nthreads;
    const int rem  = m % nthreads;
    const int extra = (tid < rem) ? tid : rem;

    *i0 = tid * base + extra;
    *i1 = *i0 + base + ((tid < rem) ? 1 : 0);
}

/* Suffisso comune del nome dei backend di CPU: rende visibile nel CSV se la
 * riga viene da una build con la specializzazione su k disattivata. */
#ifdef FORCE_GENERIC_K
#define SCHEME_A_NAME_SUFFIX "_generic"
#else
#define SCHEME_A_NAME_SUFFIX ""
#endif

#endif /* SCPA_KERNEL_SCHEME_A_CORE_H */
