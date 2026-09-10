/* Schema A parallelizzato con OpenMP: un thread per fetta di righe di Y.
 *
 * E' l'alternativa a CUDA che la traccia ammette ("una parallelizzazione
 * interna al singolo processo, in OpenMP o CUDA"), e sta dietro la stessa
 * interfaccia local_gemm degli altri backend: il codice MPI non cambia di una
 * riga, e nella campagna di misura MPI+OpenMP e' un binario come gli altri.
 *
 * ---------------------------------------------------------------------------
 * Quale ciclo si parallelizza, e perche' quello
 * ---------------------------------------------------------------------------
 * I cicli disponibili sono tre, e due si escludono subito:
 *
 *   c (0..k)  ampiezza 3..32. Meno iterazioni che core sul nodo, e sono
 *             proprio le iterazioni che stanno negli accumulatori in registro:
 *             parallelizzarle significherebbe smontare l'unica ottimizzazione
 *             del nucleo per ottenere al piu' 32 thread di lavoro ridicolo.
 *   j (0..n)  iterazioni NON indipendenti: tutte aggiornano gli stessi k
 *             accumulatori della riga i. Servirebbe una reduction su un
 *             vettore, cioe' k accumulatori privati per thread piu' una
 *             somma finale a ogni riga - sincronizzazione m volte per
 *             invocazione, contro zero.
 *   i (0..m)  iterazioni completamente indipendenti: il thread che possiede la
 *             riga i legge A[i][*] e X, e scrive SOLO Y[i][*]. Nessuna
 *             scrittura condivisa, nessuna reduction, nessun critical, nessuna
 *             barriera oltre a quella implicita di fine regione.
 *
 * Si parallelizza i. Non e' una scelta fra alternative equivalenti: e' l'unico
 * ciclo che non richiede sincronizzazione, e le lezioni sono esplicite sul
 * fatto che le sincronizzazioni costano ("synchronization is expensive",
 * "make critical regions as small as possible").
 *
 * ---------------------------------------------------------------------------
 * Una sola regione parallela per invocazione
 * ---------------------------------------------------------------------------
 * La regione parallela si apre una volta per local_gemm e contiene TUTTO il
 * lavoro dell'invocazione ("maximize size of parallel regions", "avoid
 * parallel regions in inner loops"). Il costo residuo e' quello di risvegliare
 * il team e la barriera finale: dell'ordine dei microsecondi, contro
 * un'invocazione che sui problemi della campagna dura decine di millisecondi.
 *
 * Si usa `#pragma omp parallel` con partizione esplicita delle righe e non
 * `#pragma omp parallel for`, per due motivi concreti:
 *   - il dispatch su k (i cinque kernel specializzati) avviene UNA volta per
 *     thread invece che una volta per riga o per chunk;
 *   - il thread esegue un unico ciclo su tutta la sua fetta, quindi il codice
 *     caldo e' esattamente quello del backend scalare, con gli stessi
 *     accumulatori in registro e senza il ciclo esterno che una work-sharing
 *     construct interpone. La fetta e' calcolata da scheme_a_row_range, che e'
 *     la stessa partizione a blocchi contigui di `schedule(static)`.
 *
 * Non c'e' nessuna clausola di data sharing perche' non serve: tutto cio' che
 * i thread condividono (A, X, Y, le dimensioni) e' o in sola lettura o scritto
 * su righe disgiunte, e tutto cio' che e' privato - indici e accumulatori - e'
 * dichiarato DENTRO la regione, quindi privato per costruzione. E' l'unica
 * forma che non dipende da come la versione di OpenMP del compilatore tratti
 * le variabili const sotto `default(none)`.
 *
 * ---------------------------------------------------------------------------
 * Cosa questo backend NON risolve
 * ---------------------------------------------------------------------------
 * Ogni thread percorre tutta X per ogni riga di A: il traffico in lettura su X
 * vale m_loc*n_loc*k elementi contro gli m_loc*n_loc di A, e con piu' thread
 * quella pressione si somma sulla stessa L3 condivisa. E' lo stesso problema
 * che su GPU separa cuda_warp da cuda_warp_smem, e qui lo affronta
 * omp_scheme_a_tiled. Questo backend e' il termine di paragone: la
 * parallelizzazione e basta, senza tiling.
 *
 * Nota NUMA: A viene generata (o ricevuta) dal thread master in preprocessing,
 * quindi su un nodo a piu' socket le pagine sono tutte sul nodo di memoria del
 * master ("first touch"). Con un rank MPI per socket il problema non si pone,
 * perche' la decomposizione MPI ha gia' dato a ogni processo la sua memoria;
 * con un solo rank e thread su piu' socket una parte degli accessi diventa
 * remota. E' un limite noto e va tenuto presente leggendo lo scaling oltre il
 * numero di core di un socket. */

#include "kernel/kernel.h"

#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common/util.h"
#include "kernel/scheme_a_core.h"

struct local_gemm_context {
    int m_loc, n_loc, k;
    int lda;
    int ldx, ldy;
    const scalar_t *A_loc;
    int threads;      /* dimensione del team, decisa in create e imposta a ogni
                       * invocazione: e' il piano del backend, e va riportato
                       * accanto ai tempi. */
    double t_setup;
};

/* Numero di thread del team.
 *
 * Si legge UNA volta, in create, e poi si impone con num_threads a ogni
 * invocazione. Cosi' il numero riportato nel CSV e' davvero quello che ha
 * eseguito la misura: se il runtime avesse i thread dinamici attivi
 * (OMP_DYNAMIC), un team scelto invocazione per invocazione renderebbe la
 * colonna threads una dichiarazione di intenti invece di una misura.
 *
 * Il clamp a m_loc non e' cosmetico: con piu' thread che righe i thread in
 * eccesso riceverebbero un intervallo vuoto: il tempo non cambia, ma la
 * colonna threads direbbe 20 dove i thread al lavoro sono 3. */
static int decide_threads(int m_loc)
{
    int threads = 1;

#ifdef _OPENMP
    threads = omp_get_max_threads();
    if (threads < 1)
        threads = 1;
#endif

    if (m_loc > 0 && threads > m_loc)
        threads = m_loc;

    return threads;
}

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
    local_gemm_context->threads = decide_threads(m_loc);
    local_gemm_context->t_setup = now_seconds() - t0;
    return local_gemm_context;
}

void local_gemm(local_gemm_t *local_gemm_context, const scalar_t * RESTRICT X_loc, int ldx, scalar_t * RESTRICT Y_loc_part, int ldy)
{
    const scalar_t *RESTRICT A_loc = local_gemm_context->A_loc;
    const int m_loc = local_gemm_context->m_loc,
              n_loc = local_gemm_context->n_loc,
              k = local_gemm_context->k,
              lda = local_gemm_context->lda,
              threads = local_gemm_context->threads;

    if (ldx != local_gemm_context->ldx || ldy != local_gemm_context->ldy)
        die("local_gemm: leading dimensions changed between calls "
            "(ldx %d -> %d, ldy %d -> %d)",
            local_gemm_context->ldx, ldx, local_gemm_context->ldy, ldy);

    if (m_loc <= 0)
        return;

#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
    {
        int i0, i1;

        scheme_a_row_range(m_loc, omp_get_num_threads(), omp_get_thread_num(), &i0, &i1);
        scheme_a_assign(i0, i1, 0, n_loc, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
    }
#else
    /* Compilato senza -fopenmp le direttive sparirebbero comunque, ma il
     * codice resterebbe pieno di chiamate a omp_get_*: la guardia su _OPENMP
     * e' il modo standard di tenere il file compilabile anche cosi', e rende
     * questo backend identico a scheme_a invece che rotto. */
    (void)threads;
    scheme_a_assign(0, m_loc, 0, n_loc, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
#endif
}

void local_gemm_destroy(local_gemm_t *local_gemm_context)
{
    xfree(local_gemm_context);
}

/* Come per scheme_a: su CPU il kernel coincide con l'invocazione, non c'e' un
 * trasferimento da separare, e il sentinella negativo lo dichiara. */
double local_gemm_last_compute_seconds(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1.0;
}

double local_gemm_setup_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_setup : 0.0;
}

int local_gemm_blocks_per_sm(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1;
}

/* Questo backend non stage nessuna fetta di X: ogni thread la percorre tutta.
 * E' esattamente la differenza con omp_scheme_a_tiled, ed e' giusto che nel
 * CSV la colonna resti il sentinella invece di un numero inventato. */
int local_gemm_x_rows_per_tile(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1;
}

int local_gemm_threads(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->threads : -1;
}

const char *kernel_name(void)
{
    return "omp_scheme_a" SCHEME_A_NAME_SUFFIX;
}
