/* Schema A scalare: ordine dei cicli i -> j -> c, tutto row-major, un thread.
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
 * Il ciclo caldo vive in src/kernel/scheme_a_core.h, che questo backend
 * condivide con i due backend OpenMP: qui resta soltanto il ciclo di vita del
 * contesto e la chiamata sull'intero blocco locale. E' la baseline a un thread
 * contro cui si misurano gli altri, e il fatto che eseguano LO STESSO nucleo
 * e' cio' che rende il confronto una misura della parallelizzazione. */

#include "kernel/kernel.h"

#include <stddef.h>

#include "common/util.h"
#include "kernel/scheme_a_core.h"

/* Stato del backend.
 *
 * Per lo schema A scalare non c'e' nulla da preparare: A e' gia' nella memoria
 * del processo e il kernel la legge direttamente, quindi il contesto si limita
 * a registrare forma e puntatore. Esiste comunque, e con la stessa interfaccia
 * degli altri backend, perche' e' qui che il backend CUDA tiene il puntatore
 * alla copia di A in VRAM: da quel lato create() e' una cudaMemcpy H2D che
 * deve avvenire UNA volta sola, fuori dalla regione cronometrata. */
struct local_gemm_context {
    int m_loc, n_loc, k;
    int lda;
    int ldx, ldy;
    const scalar_t *A_loc;
    double t_setup;   /* misurato davvero, anche se qui e' ~1 us: il confronto
                       * con il backend CUDA ha senso solo se lo stesso numero
                       * viene dallo stesso punto del codice in entrambi. */
};

local_gemm_t *local_gemm_create(int m_loc, int n_loc, int k, const scalar_t *A_loc, int lda, int ldx, int ldy)
{
    local_gemm_t *local_gemm_context;
    const double t0 = now_seconds();

    if (m_loc < 0 || n_loc < 0 || k < 0)
        die("local_gemm_create: invalid local block %dx%d with k=%d", m_loc, n_loc, k);
    if (lda < n_loc)
        die("local_gemm_create: lda %d is smaller than n %d", lda, n_loc);
    if (ldx < k || ldy < k)
        die("local_gemm_create: ldx %d and ldy %d must both be at least k=%d",
            ldx, ldy, k);
    if (n_loc > 0 && m_loc > 0 && A_loc == NULL)
        die("local_gemm_create: A is NULL for a non-empty %dx%d block", m_loc, n_loc);

    local_gemm_context = xmalloc(sizeof *local_gemm_context);
    local_gemm_context->m_loc = m_loc;
    local_gemm_context->n_loc = n_loc;
    local_gemm_context->k = k;
    local_gemm_context->lda = lda;
    local_gemm_context->ldx = ldx;
    local_gemm_context->ldy = ldy;
    local_gemm_context->A_loc = A_loc;
    local_gemm_context->t_setup = now_seconds() - t0;
    return local_gemm_context;
}

void local_gemm(local_gemm_t *local_gemm_context, const scalar_t * RESTRICT X_loc, int ldx, scalar_t * RESTRICT Y_loc_part, int ldy)
{
    /* Copie locali: il puntatore ad A torna a essere restrict all'interno di
     * questa funzione, cosi' i kernel specializzati ricevono la stessa
     * garanzia di non aliasing che avevano quando A era un parametro. */
    const scalar_t *RESTRICT A_loc = local_gemm_context->A_loc;
    const int m_loc = local_gemm_context->m_loc,
              n_loc = local_gemm_context->n_loc,
              k = local_gemm_context->k,
              lda = local_gemm_context->lda;

    if (ldx != local_gemm_context->ldx || ldy != local_gemm_context->ldy)
        die("local_gemm: leading dimensions changed between calls "
            "(ldx %d -> %d, ldy %d -> %d)",
            local_gemm_context->ldx, ldx, local_gemm_context->ldy, ldy);

    /* Tutte le righe, tutte le colonne, in assegnazione: e' la chiamata da cui
     * i backend OpenMP differiscono solo per l'intervallo. */
    scheme_a_assign(0, m_loc, 0, n_loc, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
}

void local_gemm_destroy(local_gemm_t *local_gemm_context)
{
    /* Nessuna risorsa esterna da rilasciare: A appartiene al chiamante.
     * Il backend CUDA fa qui la cudaFree della copia in VRAM. */
    xfree(local_gemm_context);
}

/* Su CPU il kernel E' l'invocazione: non esiste un tempo di calcolo distinto
 * da t_local, e restituire t_local qui vorrebbe dire duplicare in una colonna
 * un numero che il chiamante ha gia'. Il valore negativo dice "non applicabile"
 * e il driver lo riporta come tale, cosi' nel CSV si vede a colpo d'occhio
 * quali righe vengono da un backend con trasferimenti e quali no. */
double local_gemm_last_compute_seconds(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1.0;
}

double local_gemm_setup_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_setup : 0.0;
}

/* Occupancy e tiling in shared memory sono concetti di GPU: su CPU non esistono
 * e il sentinella negativo lo dichiara, con la stessa convenzione di
 * local_gemm_last_compute_seconds. */
int local_gemm_blocks_per_sm(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1;
}

int local_gemm_x_rows_per_tile(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1;
}

/* Qui 1 non e' un sentinella ma una misura: questo backend e' seriale per
 * costruzione ed e' il denominatore dello speedup dei backend OpenMP. Nel CSV
 * la colonna threads vale quindi 1 sulle righe della baseline e -1 su quelle
 * dei backend che non hanno il concetto di thread di CPU (CUDA). */
int local_gemm_threads(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return 1;
}

const char *kernel_name(void)
{
    return "scheme_a" SCHEME_A_NAME_SUFFIX;
}
