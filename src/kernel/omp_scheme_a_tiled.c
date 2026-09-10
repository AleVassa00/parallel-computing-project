/* Schema A con OpenMP e tiling di cache: omp_scheme_a con UNA sola variabile
 * cambiata, da dove arrivano gli elementi di X.
 *
 * ---------------------------------------------------------------------------
 * Il problema che questo backend risolve
 * ---------------------------------------------------------------------------
 * Nello schema A ogni riga di A si legge una volta sola, e questo e' ottimo per
 * A. Ma per OGNI riga di A il ciclo interno percorre tutta X: il traffico in
 * lettura vale
 *
 *     A:  m_loc * n_loc            elementi   (una volta, obbligatoria)
 *     X:  m_loc * n_loc * k        elementi   (X riletta m_loc volte)
 *
 * cioe' k volte quello di A. X non arriva in DRAM - n_loc*k scalari sono
 * pochi MB e restano in L3 - ma la banda di L3 e' condivisa da tutti i core del
 * socket, e con venti thread che la percorrono contemporaneamente diventa lei
 * il vincolo attivo, tanto piu' quanto piu' k e' grande.
 *
 * E' lo stesso identico fenomeno che sulla GPU separa cuda_warp da
 * cuda_warp_smem, e la cura e' la stessa: percorrere X a TILE, e riusare ogni
 * tile su molte righe di A prima di passare al successivo. Cambia solo dove sta
 * il tile - li' la shared memory, qui la cache L1 - perche' cambia la
 * gerarchia, non l'idea. E' anche il blocking del prodotto matrice-matrice
 * visto a lezione, applicato al caso degenere in cui una delle due dimensioni
 * (k) e' gia' piccola e quindi non va bloccata.
 *
 * ---------------------------------------------------------------------------
 * I due livelli di tile, e perche' servono entrambi
 * ---------------------------------------------------------------------------
 * L'ordine dei cicli e':
 *
 *   per ogni thread, sulla sua fetta di righe [i0,i1):
 *     for ia in i0..i1 step IB:            <- tile di righe di A e Y
 *       for j0 in 0..n_loc step JB:        <- tile di righe di X
 *         Y[ia:ia+IB, :] += A[ia:ia+IB, j0:j0+JB] * X[j0:j0+JB, :]
 *
 * JB (tile su X) esiste perche' il tile di X, JB*k scalari, deve stare in L1:
 * cosi' le IB righe di A che lo attraversano lo trovano li' invece che in L3.
 * Il riuso di ogni elemento di X passa da "una volta ogni giro completo" a IB
 * volte consecutive.
 *
 * IB (tile su A e Y) esiste perche' spezzare il ciclo su j costringe a
 * riportare i k accumulatori in Y a ogni tile e a rileggerli al tile
 * successivo. Perche' quel traffico non sostituisca semplicemente quello su X,
 * il blocco di Y - IB*k scalari - deve restare caldo in L2 per tutta la
 * scansione di X. Con il ciclo su ia ESTERNO a quello su j0, il blocco di Y si
 * carica una volta e ci resta.
 *
 * Il bilancio del traffico verso i livelli lenti diventa quindi
 *
 *     A:  m_loc * n_loc                 (invariata: e' il minimo)
 *     X:  n_loc * k * ceil(m_loc/IB)    (invece di m_loc * n_loc * k)
 *     Y:  m_loc * k                     (una volta, se IB*k sta in L2)
 *
 * e con IB dell'ordine del migliaio il termine su X smette di essere il
 * vincolo.
 *
 * ---------------------------------------------------------------------------
 * Come si scelgono IB e JB
 * ---------------------------------------------------------------------------
 * Non sono costanti: il tile pesa (righe * k * sizeof(scalar_t)) byte e k si
 * conosce solo a runtime, esattamente come TJ in cuda_warp_smem. Si fissa
 * quindi il BUDGET IN BYTE del tile e si ricava il numero di righe:
 *
 *     JB = SCPA_OMP_X_TILE_BYTES / (k * sizeof(scalar_t))
 *     IB = SCPA_OMP_Y_TILE_BYTES / (k * sizeof(scalar_t))
 *
 * I default sono 16 KiB per il tile di X (meta' di una L1 dati da 32 KiB: il
 * resto serve alle righe di A e Y che scorrono) e 256 KiB per il blocco di Y
 * (un quarto di una L2 privata da 1 MiB). Sono due ipotesi sulla macchina, non
 * verita': per questo sono knob del Makefile (OMP_X_TILE_BYTES,
 * OMP_Y_TILE_BYTES), entrano nel nome della configurazione quando non sono i
 * default, e le build coesistono come binari distinti. Il numero di righe di X
 * per tile effettivamente scelto finisce nel CSV come x_rows_per_tile, cioe'
 * nella stessa colonna in cui lo scrive cuda_warp_smem.
 *
 * ---------------------------------------------------------------------------
 * Quando conviene, e quando invece costa
 * ---------------------------------------------------------------------------
 * Il tiling non e' gratis: spezzare il ciclo su j accorcia il ciclo interno e
 * aggiunge il viavai degli accumulatori da e verso Y. Quel costo si paga
 * sempre; il guadagno arriva solo se X non ci stava gia'. La discriminante e'
 * quindi UNA sola quantita',
 *
 *     n_loc * k * sizeof(scalar_t)      (la fetta locale di X)
 *
 * confrontata con la cache di cui i thread dispongono davvero. Su una macchina
 * di sviluppo a 8 core, con m=2000 n=60000 k=32 in doppia precisione - cioe'
 * con X da 15 MB, che in nessuna cache ci sta - questo backend e' risultato
 * circa 3 volte piu' veloce di omp_scheme_a; sulla stessa macchina, con
 * m=60000 n=2000 k=32 - X da 512 KB, che invece ci sta - e' risultato circa il
 * 20% piu' lento. Il segno del confronto cambia con la forma del blocco
 * locale, non con la taglia del problema globale.
 *
 * Questo e' anche il motivo per cui i due backend restano DUE, e non uno con
 * un'euristica dentro: l'euristica avrebbe bisogno della dimensione della
 * cache, che non e' nota al codice, e nasconderebbe nel binario proprio la
 * variabile che la campagna deve misurare. Il caso degenere e' comunque
 * gestito da solo: se il budget del tile e' abbastanza grande da contenere
 * tutta X (n_loc <= JB), il ciclo in accumulo non viene mai eseguito e questo
 * backend torna a essere esattamente omp_scheme_a.
 *
 * ---------------------------------------------------------------------------
 * Nota sul false sharing
 * ---------------------------------------------------------------------------
 * Rispetto a omp_scheme_a c'e' una differenza: le righe di Y non si scrivono
 * piu' una volta sola ma ceil(n_loc/JB) volte. Sul CONFINE fra due thread, e
 * solo quando k*sizeof(scalar_t) < 64 byte (cioe' k < 8 in doppia precisione),
 * l'ultima riga di uno e la prima dell'altro cadono nella stessa cache line, e
 * quella linea rimbalza fra i due core una volta per tile invece che una volta
 * per invocazione. Sono ceil(n_loc/JB) trasferimenti per confine - qualche
 * migliaio in totale - contro gli m_loc*n_loc aggiornamenti del nucleo: e'
 * misurabile in teoria e invisibile in pratica, ma va detto e non scoperto. */

#include "kernel/kernel.h"

#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common/util.h"
#include "kernel/scheme_a_core.h"

/* Budget in byte del tile di X: deve stare nella L1 dati insieme alle righe di
 * A e di Y che le scorrono accanto. 16 KiB = meta' di una L1 da 32 KiB. */
#ifndef SCPA_OMP_X_TILE_BYTES
#define SCPA_OMP_X_TILE_BYTES 16384
#endif

/* Budget in byte del blocco di Y: deve restare caldo in L2 per tutta la
 * scansione di X. 256 KiB = un quarto di una L2 privata da 1 MiB. */
#ifndef SCPA_OMP_Y_TILE_BYTES
#define SCPA_OMP_Y_TILE_BYTES 262144
#endif

#if SCPA_OMP_X_TILE_BYTES <= 0 || SCPA_OMP_Y_TILE_BYTES <= 0
#error "SCPA_OMP_X_TILE_BYTES e SCPA_OMP_Y_TILE_BYTES devono essere positivi"
#endif

struct local_gemm_context {
    int m_loc, n_loc, k;
    int lda;
    int ldx, ldy;
    const scalar_t *A_loc;
    int threads;
    int x_rows_per_tile;   /* JB */
    int y_rows_per_tile;   /* IB */
    double t_setup;
};

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

/* Righe che stanno nel budget dato, mai zero e mai piu' del disponibile.
 * Con k == 0 non c'e' nessun elemento da tenere in cache e il tile degenera
 * nell'intero intervallo: e' il caso dei blocchi vuoti, che la validazione
 * esercita e che non deve dividere per zero. */
static int rows_in_budget(int budget_bytes, int k, int available)
{
    size_t row_bytes;
    int rows;

    if (available <= 0)
        return 0;
    if (k <= 0)
        return available;

    row_bytes = (size_t)k * sizeof(scalar_t);
    rows = (int)((size_t)budget_bytes / row_bytes);

    if (rows < 1)
        rows = 1;              /* una riga intera entra sempre: k <= 32 scalari */
    if (rows > available)
        rows = available;

    return rows;
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

    /* Il piano si calcola QUI, come su cuda_warp_smem, e per la stessa ragione:
     * k, m_loc e n_loc sono fissi per tutta l'esecuzione, il piano vale per
     * tutte le repetition, e il suo costo compare in t_setup invece che dentro
     * la regione cronometrata. */
    local_gemm_context->x_rows_per_tile = rows_in_budget(SCPA_OMP_X_TILE_BYTES, k, n_loc);
    local_gemm_context->y_rows_per_tile = rows_in_budget(SCPA_OMP_Y_TILE_BYTES, k, m_loc);

    local_gemm_context->t_setup = now_seconds() - t0;
    return local_gemm_context;
}

/* Il lavoro di un thread: la sua fetta di righe [i0,i1), percorsa a tile di IB
 * righe, e per ogni tile una scansione di X a tile di JB righe.
 *
 * Il PRIMO tile di X e' in assegnazione e gli altri in accumulo. Non e' un
 * dettaglio di stile: e' cio' che evita una passata di azzeramento di Y (m*k
 * scritture in piu' per invocazione) e che tiene il contratto di local_gemm,
 * che e' Y = A*X e non Y += A*X. Con n_loc == 0 il primo tile e' vuoto ma la
 * chiamata in assegnazione avviene lo stesso, e Y viene azzerata: e' il
 * comportamento corretto per un blocco locale senza colonne, che la suite di
 * validazione esercita. */
static void gemm_rows(int i0, int i1, int n_loc, int k, int ib_rows, int jb_rows,
                      const scalar_t *RESTRICT A_loc, int lda,
                      const scalar_t *RESTRICT X_loc, int ldx,
                      scalar_t *RESTRICT Y_loc_part, int ldy)
{
    int ia;

    for (ia = i0; ia < i1; ia += ib_rows) {
        const int ib = (i1 - ia < ib_rows) ? i1 : ia + ib_rows;
        const int first = (n_loc < jb_rows) ? n_loc : jb_rows;
        int j0;

        scheme_a_assign(ia, ib, 0, first, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);

        for (j0 = first; j0 < n_loc; j0 += jb_rows) {
            const int j1 = (n_loc - j0 < jb_rows) ? n_loc : j0 + jb_rows;
            scheme_a_accum(ia, ib, j0, j1, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        }
    }
}

void local_gemm(local_gemm_t *local_gemm_context, const scalar_t * RESTRICT X_loc, int ldx, scalar_t * RESTRICT Y_loc_part, int ldy)
{
    const scalar_t *RESTRICT A_loc = local_gemm_context->A_loc;
    const int m_loc = local_gemm_context->m_loc,
              n_loc = local_gemm_context->n_loc,
              k = local_gemm_context->k,
              lda = local_gemm_context->lda,
              threads = local_gemm_context->threads,
              ib_rows = local_gemm_context->y_rows_per_tile,
              jb_rows = local_gemm_context->x_rows_per_tile;

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
        gemm_rows(i0, i1, n_loc, k, ib_rows, jb_rows,
                  A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
    }
#else
    /* Senza -fopenmp resta il solo tiling di cache, su un thread: e' anche il
     * modo di misurare quanto del guadagno venga dal tiling e quanto dai
     * thread, senza cambiare backend. */
    (void)threads;
    gemm_rows(0, m_loc, n_loc, k, ib_rows, jb_rows,
              A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
#endif
}

void local_gemm_destroy(local_gemm_t *local_gemm_context)
{
    xfree(local_gemm_context);
}

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

/* Stessa colonna del CSV in cui cuda_warp_smem scrive le righe di X per tile
 * di shared memory: qui il tile e' in L1 invece che in shared, ma il numero
 * significa la stessa cosa e serve allo stesso scopo, cioe' rendere leggibile
 * una campagna sul tiling. IB non ha una colonna propria ed e' ricostruibile:
 * vale OMP_Y_TILE_BYTES / (k*sizeof(scalar_t)) limitato a m_loc, e il budget
 * compare nel nome del kernel quando non e' quello di default. */
int local_gemm_x_rows_per_tile(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->x_rows_per_tile : -1;
}

int local_gemm_threads(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->threads : -1;
}

/* Come i backend CUDA: i knob che non sono al default entrano nel nome, cosi'
 * nel CSV le righe di uno sweep sui budget restano distinguibili fra loro. */
#define SCPA_STR_(x) #x
#define SCPA_STR(x)  SCPA_STR_(x)

#if SCPA_OMP_X_TILE_BYTES == 16384
#define SCPA_XTILE_SUFFIX ""
#else
#define SCPA_XTILE_SUFFIX "(xtile" SCPA_STR(SCPA_OMP_X_TILE_BYTES) ")"
#endif

#if SCPA_OMP_Y_TILE_BYTES == 262144
#define SCPA_YTILE_SUFFIX ""
#else
#define SCPA_YTILE_SUFFIX "(ytile" SCPA_STR(SCPA_OMP_Y_TILE_BYTES) ")"
#endif

const char *kernel_name(void)
{
    return "omp_scheme_a_tiled" SCHEME_A_NAME_SUFFIX
           SCPA_XTILE_SUFFIX SCPA_YTILE_SUFFIX;
}
