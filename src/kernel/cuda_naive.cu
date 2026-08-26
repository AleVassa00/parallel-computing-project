/* Backend CUDA - versione naive.  E' la BASELINE, non il kernel definitivo.
 *
 * Serve a due cose, in quest'ordine:
 *   1. validare la pipeline (build con nvcc, ciclo di vita a tre fasi,
 *      trasferimenti, misura con i cudaEvent) prima di scrivere il kernel che
 *      conta, cosi' che quando quello arrivera' l'unica cosa nuova sara' lui;
 *   2. dare un numero di partenza contro cui misurare il guadagno delle
 *      versioni successive. Uno speedup senza baseline non e' un risultato.
 *
 * ---------------------------------------------------------------------------
 * Mappatura: un thread per ELEMENTO di Y
 * ---------------------------------------------------------------------------
 * Il thread t calcola Y[i][c] con  i = t / k  e  c = t % k, scorrendo tutta la
 * dimensione j:
 *
 *     acc = 0;  for j: acc += A[i][j] * X[j][c];   Y[i][c] = acc;
 *
 * Due conseguenze, entrambe volute perche' e' proprio quello che il kernel
 * successivo dovra' battere:
 *
 *  - il RIUSO DI A IN REGISTRO E' PERSO. Sullo schema A di CPU la riga di A si
 *    legge una volta e si riusa k volte dagli accumulatori; qui ogni thread
 *    tiene UN solo accumulatore, quindi la stessa riga di A viene richiesta k
 *    volte, una per ciascuno dei k thread della riga. Le richieste sono
 *    simultanee e la cache le assorbe quasi tutte, ma il lavoro c'e';
 *  - la COALESCENZA e' parziale. Thread consecutivi hanno c consecutivo, e
 *    leggono quindi X[j][c] contigui: per k = 3 sono 24 byte per riga di X, per
 *    k = 32 sono 256 byte, cioe' da meno di una transazione a due transazioni
 *    piene. Ogni gruppo di k thread legge poi lo STESSO A[i][j].
 *
 * La versione warp-shuffle di M4.3 rovescia entrambe le cose: distribuisce il
 * ciclo su j fra le 32 corsie di un warp (lane l prende j = l, l+32, ...), cosi'
 * corsie consecutive leggono elementi consecutivi di A - coalescenza piena - e
 * ogni corsia tiene tutti e k gli accumulatori in registro, riducendoli alla
 * fine con __shfl_down_sync. Qui non si fa: e' la baseline.
 *
 * ---------------------------------------------------------------------------
 * Dove finiscono i trasferimenti
 * ---------------------------------------------------------------------------
 * A non cambia mai fra un'invocazione e l'altra, X e Y si': l'H2D di A sta in
 * local_gemm_create (una volta sola, preprocessing), mentre H2D di X e D2H di Y
 * sono per forza dentro local_gemm, perche' X arriva dal MPI_Bcast e Y deve
 * tornare al MPI_Reduce. La consegna consente di escludere i trasferimenti dal
 * tempo T: per questo il kernel viene cronometrato a parte con i cudaEvent e
 * pubblicato da local_gemm_last_compute_seconds. Un lancio e' asincrono, quindi
 * l'orologio dell'host misurerebbe l'accodamento, non l'esecuzione: gli event
 * sono l'unico strumento corretto. */

#include <cuda_runtime.h>

#include <stdlib.h>

#include "kernel/kernel.h"
#include "common/util.h"

/* Ogni chiamata CUDA puo' fallire e nessuna lo segnala da sola. Senza questa
 * macro un errore si manifesterebbe molto piu' avanti, come risultato
 * sbagliato, e non come messaggio. die() abbatte l'intero job MPI: e' cio' che
 * serve, perche' un rank morto da solo lascerebbe gli altri fermi in una
 * collettiva. */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess)                                              \
            die("CUDA error at %s:%d: %s: %s", __FILE__, __LINE__,            \
                cudaGetErrorName(err_), cudaGetErrorString(err_));            \
    } while (0)

/* 256 thread per blocco: 8 warp, divisore di 1024, valore neutro che non
 * pregiudica l'occupancy su Turing. Non e' un parametro da ottimizzare qui:
 * lo diventa nel kernel di M4.3. */
#define BLOCK_THREADS 256

#define BYTES_PER_GIB 1073741824.0

static __global__ void naive_kernel(int m, int n, int k,
                                    const scalar_t *__restrict__ A, int lda,
                                    const scalar_t *__restrict__ X, int ldx,
                                    scalar_t *__restrict__ Y, int ldy)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)m * (long long)k;
    int i, c, j;
    const scalar_t *arow;
    scalar_t acc = (scalar_t)0;

    /* La griglia e' arrotondata al blocco: gli ultimi thread non hanno lavoro. */
    if (tid >= total)
        return;

    i = (int)(tid / k);
    c = (int)(tid % k);
    arow = A + (size_t)i * (size_t)lda;

    for (j = 0; j < n; j++)
        acc += arow[j] * X[(size_t)j * (size_t)ldx + c];

    /* Assegnazione, non accumulo: e' il contratto di local_gemm. Con n == 0 il
     * ciclo non gira e qui si scrive 0, che e' il risultato corretto per un
     * blocco di colonne vuoto - la MPI_Reduce che segue lo sommera' agli altri
     * contributi della stessa riga della griglia. */
    Y[(size_t)i * (size_t)ldy + c] = acc;
}

/* Stato del backend: le tre copie in VRAM, i due event e i tempi. */
struct local_gemm_ctx {
    int m, n, k;
    int lda;
    int ldx, ldy;        /* fissate alla prima invocazione (-1 = non ancora) */
    scalar_t *dA;
    scalar_t *dX;
    scalar_t *dY;
    cudaEvent_t ev_start, ev_stop;
    int device;
    double t_setup;      /* preprocessing: contesto + cudaMalloc + H2D di A */
    double t_last;       /* solo kernel, ultima invocazione (< 0 = mai) */
};

/* Con MPS disattivato piu' rank sullo stesso device funzionano ma si
 * alternano; su un nodo con piu' schede conviene comunque che rank diversi ne
 * prendano una ciascuno. Il rank LOCALE lo pubblica gia' il lanciatore in una
 * variabile d'ambiente, cosi' questo file non deve dipendere da MPI. */
static int pick_device(void)
{
    const char *env;
    int ndev = 0;
    long id;

    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1)
        die("cuda_naive: no CUDA device available");

    env = getenv("SCPA_CUDA_DEVICE");            /* override esplicito */
    if (env == NULL) env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");   /* OpenMPI */
    if (env == NULL) env = getenv("MV2_COMM_WORLD_LOCAL_RANK");    /* MVAPICH */
    if (env == NULL) env = getenv("SLURM_LOCALID");                /* Slurm   */

    id = (env != NULL) ? strtol(env, NULL, 10) : 0;
    if (id < 0)
        id = 0;
    return (int)(id % ndev);
}

static size_t nonzero(size_t bytes)
{
    /* cudaMalloc(0) non ha un comportamento su cui valga la pena contare, e un
     * blocco locale vuoto e' legittimo (griglia piu' grande della matrice).
     * Si alloca il minimo, come fa xmalloc sull'host. */
    return (bytes != 0) ? bytes : 1;
}

local_gemm_t *local_gemm_create(int m, int n, int k,
                                const scalar_t *A, int lda)
{
    local_gemm_t *ctx;
    size_t bytes_A, need, free_b = 0, total_b = 0;
    const double t0 = now_seconds();

    /* Stessi controlli del backend scalare: l'interfaccia e' una sola, e i suoi
     * invarianti non possono dipendere da chi la implementa. */
    if (m < 0 || n < 0 || k < 0)
        die("local_gemm_create: invalid local block %dx%d with k=%d", m, n, k);
    if (lda < n)
        die("local_gemm_create: lda %d is smaller than n %d", lda, n);
    if (n > 0 && m > 0 && A == NULL)
        die("local_gemm_create: A is NULL for a non-empty %dx%d block", m, n);

    ctx = (local_gemm_t *)xmalloc(sizeof *ctx);
    ctx->m = m;
    ctx->n = n;
    ctx->k = k;
    ctx->lda = lda;
    ctx->ldx = -1;
    ctx->ldy = -1;
    ctx->dA = NULL;
    ctx->dX = NULL;
    ctx->dY = NULL;
    ctx->t_last = -1.0;

    ctx->device = pick_device();
    CUDA_CHECK(cudaSetDevice(ctx->device));

    /* La creazione del contesto CUDA e' pigra e costa da un decimo di secondo a
     * qualche secondo. cudaFree(0) la forza QUI, dove siamo in preprocessing;
     * senza, il conto verrebbe presentato alla prima local_gemm, cioe' dentro la
     * regione cronometrata (e con --warmup 0 dentro la prima repetition). */
    CUDA_CHECK(cudaFree(0));

    bytes_A = (size_t)m * (size_t)lda * sizeof(scalar_t);

    /* A e' l'oggetto grande: 40000x40000 in double sono 12.8 GiB contro i 15.5
     * GiB della Quadro RTX 5000. Meglio un messaggio che dice quanto manca che
     * un "out of memory" generico a meta' campagna. Nella stima si contano
     * anche X e Y, che qui non sono ancora dimensionati: ldx e ldy arrivano
     * dalla prima invocazione e valgono almeno k. */
    need = bytes_A
         + (size_t)n * (size_t)k * sizeof(scalar_t)
         + (size_t)m * (size_t)k * sizeof(scalar_t);
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (need + (64u << 20) > free_b)
        die("cuda_naive: device %d has %.2f GiB free out of %.2f GiB, "
            "but the local block needs about %.2f GiB "
            "(A %dx%d, X %dx%d, Y %dx%d in %s): use more MPI processes "
            "or a smaller M/N",
            ctx->device, (double)free_b / BYTES_PER_GIB,
            (double)total_b / BYTES_PER_GIB, (double)need / BYTES_PER_GIB,
            m, n, n, k, m, k, SCALAR_NAME);

    CUDA_CHECK(cudaMalloc((void **)&ctx->dA, nonzero(bytes_A)));
    if (bytes_A > 0)
        CUDA_CHECK(cudaMemcpy(ctx->dA, A, bytes_A, cudaMemcpyHostToDevice));

    /* Gli event vanno creati una volta sola: crearli e distruggerli a ogni
     * invocazione sarebbe una chiamata al driver dentro la misura. */
    CUDA_CHECK(cudaEventCreate(&ctx->ev_start));
    CUDA_CHECK(cudaEventCreate(&ctx->ev_stop));

    ctx->t_setup = now_seconds() - t0;
    return ctx;
}

/* ldx e ldy non sono parametri di local_gemm_create - li porta l'invocazione -
 * quindi i buffer di X e Y si allocano alla prima chiamata e non piu'. Nel
 * driver quella chiamata e' un warm-up (--warmup vale 2 per default), percio' la
 * allocazione resta fuori dalle repetition cronometrate; il suo costo viene
 * comunque sommato a t_setup, cosi' non sparisce dai dati. */
static void ensure_xy(local_gemm_t *ctx, int ldx, int ldy)
{
    double t0;

    if (ctx->dX != NULL) {
        /* Il layout e' calcolato una volta sola in layout_init: se cambiasse
         * qui, i buffer di device avrebbero la dimensione sbagliata. Meglio
         * fermarsi che leggere fuori. */
        if (ldx != ctx->ldx || ldy != ctx->ldy)
            die("cuda_naive: leading dimensions changed between calls "
                "(ldx %d -> %d, ldy %d -> %d)", ctx->ldx, ldx, ctx->ldy, ldy);
        return;
    }

    if (ldx < ctx->k || ldy < ctx->k)
        die("cuda_naive: ldx %d and ldy %d must both be at least k=%d",
            ldx, ldy, ctx->k);

    t0 = now_seconds();
    ctx->ldx = ldx;
    ctx->ldy = ldy;
    CUDA_CHECK(cudaMalloc((void **)&ctx->dX,
                          nonzero((size_t)ctx->n * (size_t)ldx * sizeof(scalar_t))));
    CUDA_CHECK(cudaMalloc((void **)&ctx->dY,
                          nonzero((size_t)ctx->m * (size_t)ldy * sizeof(scalar_t))));
    /* Il kernel scrive solo le colonne [0, k): se ldy > k il resto del buffer
     * verrebbe ricopiato sull'host cosi' com'e'. Azzerarlo una volta evita di
     * riportare indietro memoria non inizializzata. */
    CUDA_CHECK(cudaMemset(ctx->dY, 0,
                          nonzero((size_t)ctx->m * (size_t)ldy * sizeof(scalar_t))));
    ctx->t_setup += now_seconds() - t0;
}

void local_gemm(local_gemm_t *ctx,
                const scalar_t *SCPA_RESTRICT X, int ldx,
                scalar_t *SCPA_RESTRICT Y, int ldy)
{
    const int m = ctx->m, n = ctx->n, k = ctx->k;
    const long long threads = (long long)m * (long long)k;
    float ms = 0.0f;

    ensure_xy(ctx, ldx, ldy);

    /* H2D di X: una fetta n x k, cioe' pochi MiB. E' dentro l'invocazione
     * perche' X e' il risultato del MPI_Bcast appena concluso. */
    if (n > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(ctx->dX, X,
                              (size_t)n * (size_t)ldx * sizeof(scalar_t),
                              cudaMemcpyHostToDevice));

    /* Gli event delimitano il SOLO kernel: quello che sta fra i due record e'
     * cio' che finisce in t_kernel e quindi in gflops_kernel. */
    CUDA_CHECK(cudaEventRecord(ctx->ev_start, 0));
    if (threads > 0) {
        const int blocks = (int)((threads + BLOCK_THREADS - 1) / BLOCK_THREADS);
        naive_kernel<<<blocks, BLOCK_THREADS>>>(m, n, k,
                                                ctx->dA, ctx->lda,
                                                ctx->dX, ldx,
                                                ctx->dY, ldy);
        /* Un lancio fallito non lancia eccezioni: si legge cosi'. */
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(ctx->ev_stop, 0));

    /* D2H di Y. Essendo sullo stream di default con memoria paginabile, questa
     * copia e' anche il punto di sincronizzazione: quando ritorna, il kernel ha
     * finito e Ypart e' pronta per la MPI_Reduce. */
    if (m > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(Y, ctx->dY,
                              (size_t)m * (size_t)ldy * sizeof(scalar_t),
                              cudaMemcpyDeviceToHost));

    /* Con blocco locale vuoto non c'e' stata nessuna copia a sincronizzare:
     * l'attesa esplicita sull'event serve comunque, ed e' gratuita quando il
     * lavoro e' gia' concluso. */
    CUDA_CHECK(cudaEventSynchronize(ctx->ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ctx->ev_start, ctx->ev_stop));
    ctx->t_last = (double)ms * 1.0e-3;
}

void local_gemm_destroy(local_gemm_t *ctx)
{
    if (ctx == NULL)
        return;

    /* In chiusura gli errori non si possono piu' gestire in modo utile: si
     * libera quello che c'e' e si esce. */
    if (ctx->dA != NULL) cudaFree(ctx->dA);
    if (ctx->dX != NULL) cudaFree(ctx->dX);
    if (ctx->dY != NULL) cudaFree(ctx->dY);
    cudaEventDestroy(ctx->ev_start);
    cudaEventDestroy(ctx->ev_stop);
    xfree(ctx);
}

double local_gemm_last_compute_seconds(const local_gemm_t *ctx)
{
    return (ctx != NULL) ? ctx->t_last : -1.0;
}

double local_gemm_setup_seconds(const local_gemm_t *ctx)
{
    return (ctx != NULL) ? ctx->t_setup : 0.0;
}

const char *kernel_name(void)
{
    return "cuda_naive";
}
