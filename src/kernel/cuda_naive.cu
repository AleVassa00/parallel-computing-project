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
 * La versione warp-shuffle di M4.2 rovescia entrambe le cose: distribuisce il
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

#include <stdio.h>
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

/* Thread per blocco.
 *
 * 256 e' il default e non e' un numero arbitrario: sono 8 warp, ed e' un
 * DIVISORE di 1024, che su Turing (sm_75) e' il massimo di thread residenti per
 * SM. Con blocchi da 256 ne stanno 4 per SM e l'SM e' pieno; con 192 o 384 -
 * altre due scelte comuni - ne starebbero 5 e 2, cioe' 960 e 768 thread su
 * 1024. Deve inoltre restare un multiplo di 32: un blocco che non lo e' spreca
 * le corsie dell'ultimo warp in OGNI blocco, non solo nell'ultimo.
 *
 * Qui i thread non cooperano (niente shared, niente __syncthreads), quindi la
 * dimensione del blocco non cambia il risultato: cambia solo come lo scheduler
 * riempie gli SM. Il valore si sostituisce dal Makefile con BLOCK=<n>, che
 * definisce SCPA_BLOCK_THREADS, cosi' la sensibilita' del kernel a questo
 * parametro si MISURA invece di darla per buona. Ogni valore produce un binario
 * con nome proprio e un kernel_name() distinto: le righe di uno sweep restano
 * distinguibili nel CSV. */
#ifndef SCPA_BLOCK_THREADS
#define SCPA_BLOCK_THREADS 256
#endif
#if SCPA_BLOCK_THREADS < 32 || SCPA_BLOCK_THREADS > 1024 || (SCPA_BLOCK_THREADS % 32) != 0
#error "SCPA_BLOCK_THREADS deve essere un multiplo di 32 compreso fra 32 e 1024"
#endif
#define BLOCK_THREADS SCPA_BLOCK_THREADS

#define BYTES_PER_GIB 1073741824.0

static __global__ void naive_kernel(int m, int n, int k, const scalar_t *__restrict__ A, int lda, const scalar_t *__restrict__ X, int ldx, scalar_t *__restrict__ Y, int ldy)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)m * (long long)k;

    int i, c, j;

    const scalar_t *arow;

    scalar_t acc = (scalar_t)0;

    /* La griglia e' arrotondata al blocco: gli ultimi thread non hanno lavoro. */
    if (tid >= total)
        return;

    i = (int)(tid / k); // riga di Y di cui si è responsabili -> corrisponde anche alla riga di A
    c = (int)(tid % k); // colonna di Y di cui si è responsabili -> corrisponde anche alla colonna di X

    arow = A + (size_t)i * (size_t)lda; // selezioniamo la riga corrispondente di A

    /* Prodotto riga per colonna
     * Una volta fissata la riga di A e la colonna di X
     * viene calcolato il prodotto riga per colonna, scorrendo gli elementi
     * con l'indice j
     * A[i][j] * X[j][c]
     */
    for (j = 0; j < n; j++)
        acc += arow[j] * X[(size_t)j * (size_t)ldx + c];

    /* Assegnazione, non accumulo: e' il contratto di local_gemm. Con n == 0 il
     * ciclo non gira e qui si scrive 0, che e' il risultato corretto per un
     * blocco di colonne vuoto - la MPI_Reduce che segue lo sommera' agli altri
     * contributi della stessa riga della griglia. */
    Y[(size_t)i * (size_t)ldy + c] = acc;
}

/* Stato del backend: le tre copie in VRAM, i due event e i tempi. */
struct local_gemm_context {
    int m, n, k;
    int lda;
    int ldx, ldy;
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
    const char *env, *local_rank_env, *local_size_env;
    int ndev = 0;
    long id, local_rank, local_size;

    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1)
        die("cuda_naive: no CUDA device available");

    local_rank_env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");       /* OpenMPI */
    if (local_rank_env == NULL)
        local_rank_env = getenv("MV2_COMM_WORLD_LOCAL_RANK");    /* MVAPICH */
    if (local_rank_env == NULL)
        local_rank_env = getenv("SLURM_LOCALID");                /* Slurm   */

    local_rank = (local_rank_env != NULL) ? strtol(local_rank_env, NULL, 10) : 0;
    if (local_rank < 0)
        local_rank = 0;

    local_size_env = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
    if (local_size_env == NULL)
        local_size_env = getenv("MV2_COMM_WORLD_LOCAL_SIZE");
    if (local_size_env == NULL)
        local_size_env = getenv("SLURM_NTASKS_PER_NODE");
    local_size = (local_size_env != NULL) ? strtol(local_size_env, NULL, 10) : 1;

    if (local_rank == 0 && local_size > ndev)
        fprintf(stderr,
                "warning: %ld local MPI ranks share %d CUDA device(s); "
                "correctness is preserved, but official/kernel GPU timing "
                "does not represent one-rank-per-GPU throughput\n",
                local_size, ndev);

    env = getenv("SCPA_CUDA_DEVICE");            /* override esplicito */
    id = (env != NULL) ? strtol(env, NULL, 10) : local_rank;

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

local_gemm_t *local_gemm_create(int m_loc, int n_loc, int k, const scalar_t *A_loc, int lda, int ldx, int ldy)
{
    local_gemm_t *local_gemm_context;
    size_t bytes_A, bytes_X, bytes_Y, need, free_b = 0, total_b = 0;
    const double t0 = now_seconds();

    /* Stessi controlli del backend scalare: l'interfaccia e' una sola, e i suoi
     * invarianti non possono dipendere da chi la implementa. */
    if (m_loc < 0 || n_loc < 0 || k < 0)
        die("local_gemm_create: invalid local block %dx%d with k=%d", m_loc, n_loc, k);
    if (lda < n_loc)
        die("local_gemm_create: lda %d is smaller than n %d", lda, n_loc);
    if (ldx < k || ldy < k)
        die("local_gemm_create: ldx %d and ldy %d must both be at least k=%d",
            ldx, ldy, k);
    if (n_loc > 0 && m_loc > 0 && A_loc == NULL)
        die("local_gemm_create: A is NULL for a non-empty %dx%d block", m_loc, n_loc);

    local_gemm_context = (local_gemm_t *)xmalloc(sizeof *local_gemm_context);
    local_gemm_context->m = m_loc;
    local_gemm_context->n = n_loc;
    local_gemm_context->k = k;
    local_gemm_context->lda = lda;
    local_gemm_context->ldx = ldx;
    local_gemm_context->ldy = ldy;
    local_gemm_context->dA = NULL;
    local_gemm_context->dX = NULL;
    local_gemm_context->dY = NULL;
    local_gemm_context->t_last = -1.0;

    local_gemm_context->device = pick_device();
    CUDA_CHECK(cudaSetDevice(local_gemm_context->device));

    /* I runtime recenti inizializzano il contesto gia' in cudaSetDevice;
     * cudaFree(0) forza l'inizializzazione nei runtime precedenti ed e' un
     * no-op sicuro negli altri. In entrambi i casi il costo resta QUI, nel
     * preprocessing, e non nella prima local_gemm con --warmup 0. */
    CUDA_CHECK(cudaFree(0));

    bytes_A = (size_t)m_loc * (size_t)lda * sizeof(scalar_t);
    bytes_X = (size_t)n_loc * (size_t)ldx * sizeof(scalar_t);
    bytes_Y = (size_t)m_loc * (size_t)ldy * sizeof(scalar_t);

    /* A e' l'oggetto grande: 40000x40000 in double sono 12.8 GiB contro i 15.5
     * GiB della Quadro RTX 5000. Meglio un messaggio che dice quanto manca che
     * un "out of memory" generico a meta' campagna. La stima usa i layout
     * effettivi di A, X e Y ricevuti dal driver. */
    need = bytes_A + bytes_X + bytes_Y;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (need + (64u << 20) > free_b)
        die("cuda_naive: device %d has %.2f GiB free out of %.2f GiB, "
            "but the local block needs about %.2f GiB "
            "(A %dx%d, X %dx%d, Y %dx%d in %s): use more MPI processes "
            "or a smaller M/N",
            local_gemm_context->device, (double)free_b / BYTES_PER_GIB,
            (double)total_b / BYTES_PER_GIB, (double)need / BYTES_PER_GIB,
            m_loc, n_loc, n_loc, k, m_loc, k, SCALAR_NAME);

    CUDA_CHECK(cudaMalloc((void **)&local_gemm_context->dA, nonzero(bytes_A)));
    if (bytes_A > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dA, A_loc, bytes_A, cudaMemcpyHostToDevice));

    /* Anche X e Y vengono dimensionate qui: nessuna cudaMalloc puo' quindi
     * cadere nella prima repetition, neppure con --warmup 0. */
    CUDA_CHECK(cudaMalloc((void **)&local_gemm_context->dX, nonzero(bytes_X)));
    CUDA_CHECK(cudaMalloc((void **)&local_gemm_context->dY, nonzero(bytes_Y)));
    CUDA_CHECK(cudaMemset(local_gemm_context->dY, 0, nonzero(bytes_Y)));

    /* Gli event vanno creati una volta sola: crearli e distruggerli a ogni
     * invocazione sarebbe una chiamata al driver dentro la misura. */
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_start));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_stop));

    /* Le copie H2D da memoria paginabile possono ritornare prima che il DMA
     * sia terminato. Questa sincronizzazione garantisce che tutto il setup sia
     * davvero concluso prima di uscire dalla regione di preprocessing. */
    CUDA_CHECK(cudaDeviceSynchronize());

    local_gemm_context->t_setup = now_seconds() - t0;
    return local_gemm_context;
}

void local_gemm(local_gemm_t *local_gemm_context, const scalar_t *RESTRICT X_loc, int ldx, scalar_t *RESTRICT Y_loc_part, int ldy)
{
    const int m_loc = local_gemm_context->m, n_loc = local_gemm_context->n, k = local_gemm_context->k;
    const long long threads = (long long)m_loc * (long long)k;
    float ms = 0.0f;

    // Guardia: le leading dimension non possono cambiare fra le chiamate
    if (ldx != local_gemm_context->ldx || ldy != local_gemm_context->ldy)
        die("cuda_naive: leading dimensions changed between calls "
            "(ldx %d -> %d, ldy %d -> %d)",
            local_gemm_context->ldx, ldx, local_gemm_context->ldy, ldy);

    /* H2D di X_loc: una "fetta" n_loc x k, cioe' pochi MiB. E' dentro l'invocazione
     * perche' X_loc e' il risultato del MPI_Bcast appena concluso. */
    if (n_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dX, X_loc, (size_t)n_loc * (size_t)ldx * sizeof(scalar_t), cudaMemcpyHostToDevice));

    /* Gli event delimitano il SOLO kernel: quello che sta fra i due record e'
     * cio' che finisce in t_kernel e quindi in gflops_kernel. */
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_start, 0));
    if (threads > 0) {
        const int blocks = (int)((threads + BLOCK_THREADS - 1) / BLOCK_THREADS);
        naive_kernel<<<blocks, BLOCK_THREADS>>>(m_loc, n_loc, k, local_gemm_context->dA, local_gemm_context->lda, local_gemm_context->dX, ldx, local_gemm_context->dY, ldy);
        /* Un lancio fallito non lancia eccezioni: si legge cosi'. */
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_stop, 0));

    /* D2H di Y.
     * Essendo sullo stream di default con memoria paginabile, questa
     * copia e' anche il punto di sincronizzazione: quando ritorna, il kernel ha
     * finito e Ypart e' pronta per la MPI_Reduce. */
    if (m_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(Y_loc_part, local_gemm_context->dY, (size_t)m_loc * (size_t)ldy * sizeof(scalar_t), cudaMemcpyDeviceToHost));

    /* Con blocco locale vuoto non c'e' stata nessuna copia a sincronizzare:
     * l'attesa esplicita sull'event serve comunque, ed e' gratuita quando il
     * lavoro e' gia' concluso. */
    CUDA_CHECK(cudaEventSynchronize(local_gemm_context->ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_start, local_gemm_context->ev_stop));
    local_gemm_context->t_last = (double)ms * 1.0e-3;
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

/* Il nome porta la dimensione del blocco quando non e' quella di default: nel
 * CSV le righe di uno sweep su BLOCK devono restare distinguibili fra loro. */
#define SCPA_STR_(x) #x
#define SCPA_STR(x)  SCPA_STR_(x)

#if SCPA_BLOCK_THREADS == 256
#define SCPA_BLK_SUFFIX ""
#else
#define SCPA_BLK_SUFFIX "(blk" SCPA_STR(SCPA_BLOCK_THREADS) ")"
#endif

const char *kernel_name(void)
{
    return "cuda_naive" SCPA_BLK_SUFFIX;
}
