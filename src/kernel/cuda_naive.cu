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
 * definisce BLOCK_THREADS, cosi' la sensibilita' del kernel a questo
 * parametro si MISURA invece di darla per buona. Ogni valore produce un binario
 * con nome proprio e un kernel_name() distinto: le righe di uno sweep restano
 * distinguibili nel CSV. */
#ifndef BLOCK_THREADS
#define BLOCK_THREADS 256
#endif
#if BLOCK_THREADS < 32 || BLOCK_THREADS > 1024 || (BLOCK_THREADS % 32) != 0
#error "BLOCK_THREADS deve essere un multiplo di 32 compreso fra 32 e 1024"
#endif

/* Il server di dipartimento su cui la consegna richiede di misurare ha una
 * sola GPU: non c'e' nessun device da scegliere, e una logica di selezione
 * basata sul rank locale MPI sarebbe codice che non puo' mai fare nulla.
 * Tutti i rank sullo stesso nodo usano quindi il device 0.
 *
 * Conseguenza da tenere presente nel leggere le misure: con piu' rank MPI per
 * GPU i contesti si alternano sulla scheda (MPS non attivo), quindi t_kernel
 * comprende anche il tempo in cui il contesto di questo rank e' sospeso a
 * favore di un altro. Per caratterizzare il kernel in se' va usato un rank
 * singolo; con piu' rank il numero resta valido come throughput AGGREGATO
 * della GPU sul problema globale, non come tempo di una GPU dedicata. */
#define CUDA_DEVICE_ID 0

#define BYTES_PER_GIB 1073741824.0

static __global__ void naive_kernel(int m_loc, int n_loc, int k, const scalar_t *__restrict__ A, int lda, const scalar_t *__restrict__ X, int ldx, scalar_t *__restrict__ Y, int ldy)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)m_loc * (long long)k;

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
    for (j = 0; j < n_loc; j++)
        acc += arow[j] * X[(size_t)j * (size_t)ldx + c];

    /* Assegnazione, non accumulo: e' il contratto di local_gemm. Con n == 0 il
     * ciclo non gira e qui si scrive 0, che e' il risultato corretto per un
     * blocco di colonne vuoto - la MPI_Reduce che segue lo sommera' agli altri
     * contributi della stessa riga della griglia. */
    Y[(size_t)i * (size_t)ldy + c] = acc;
}

/* Stato del backend: le tre copie in VRAM, i due event e i tempi. */
struct local_gemm_context {
    int m_loc, n_loc, k;
    int lda;
    int ldx, ldy;
    scalar_t *dA_loc;
    scalar_t *dX_loc;
    scalar_t *dY_loc_part;
    /* Tre coppie di event, non una: la consegna esclude dal tempo ufficiale i
     * trasferimenti da e verso la scheda ma consente di misurarli a parte, e
     * per farlo servono confini propri. Gli event sono l'unico strumento
     * corretto: la D2H parte quando il kernel ha finito, quindi un cronometro
     * sull'host le attribuirebbe anche l'attesa del calcolo. */
    cudaEvent_t ev_h2d_X_start, ev_h2d_X_stop;
    cudaEvent_t ev_kernel_start, ev_kernel_stop;
    cudaEvent_t ev_d2h_Y_start, ev_d2h_Y_stop;
    double t_setup;      /* preprocessing: contesto + cudaMalloc + H2D di A */
    double t_last;       /* solo kernel, ultima invocazione (< 0 = mai) */
    double t_last_h2d_X;   /* H2D di X, ultima invocazione (< 0 = mai) */
    double t_last_d2h_Y;   /* D2H di Y, ultima invocazione (< 0 = mai) */
    /* Scomposizione di t_setup: contesto, VRAM e la sola copia H2D di A.
     * Sommate valgono meno del totale, che comprende anche i controlli sugli
     * argomenti e la creazione degli event. */
    double t_setup_device_init;
    double t_setup_device_alloc;
    double t_setup_h2d_A;
    /* Byte che attraversano davvero il PCIe, per convertire i tempi in banda. */
    size_t bytes_h2d_A;
    size_t bytes_h2d_X_per_call;
    size_t bytes_d2h_Y_per_call;
};

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
    cudaError_t err;
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
    local_gemm_context->m_loc = m_loc;
    local_gemm_context->n_loc = n_loc;
    local_gemm_context->k = k;
    local_gemm_context->lda = lda;
    local_gemm_context->ldx = ldx;
    local_gemm_context->ldy = ldy;
    local_gemm_context->dA_loc = NULL;
    local_gemm_context->dX_loc = NULL;
    local_gemm_context->dY_loc_part = NULL;
    local_gemm_context->t_last = -1.0;
    local_gemm_context->t_last_h2d_X = -1.0;
    local_gemm_context->t_last_d2h_Y = -1.0;

    /* Voce 1 del preprocessing: creazione del contesto CUDA. E' un costo
     * fisso, indipendente dalla taglia, e va tenuto separato dalle altre due
     * per non farlo passare per un costo di trasferimento. */
    const double t_device_init_start = now_seconds();
    CUDA_CHECK(cudaSetDevice(CUDA_DEVICE_ID));

    /* I runtime recenti inizializzano il contesto gia' in cudaSetDevice;
     * cudaFree(0) forza l'inizializzazione nei runtime precedenti ed e' un
     * no-op sicuro negli altri. In entrambi i casi il costo resta QUI, nel
     * preprocessing, e non nella prima local_gemm con --warmup 0. */
    CUDA_CHECK(cudaFree(0));
    local_gemm_context->t_setup_device_init = now_seconds() - t_device_init_start;

    bytes_A = (size_t)m_loc * (size_t)lda * sizeof(scalar_t);
    bytes_X = (size_t)n_loc * (size_t)ldx * sizeof(scalar_t);
    bytes_Y = (size_t)m_loc * (size_t)ldy * sizeof(scalar_t);

    /* A e' l'oggetto grande: 40000x40000 in double sono 11.9 GiB sui 16 della
     * Quadro RTX 5000, e con piu' rank per GPU il totale va moltiplicato per il
     * numero di rank locali. need somma i layout EFFETTIVI (leading dimension,
     * non forme logiche) e serve solo a rendere leggibile la diagnosi quando
     * un'allocazione fallisce: non e' piu' un valore su cui si decide. */
    need = bytes_A + bytes_X + bytes_Y;

    /* Capienza in VRAM: si VERIFICA TENTANDO, non stimando prima.
     *
     * Il driver e' l'unico a sapere quanto costano davvero allineamento e
     * frammentazione, e - su questo server, dove piu' rank MPI condividono
     * l'unica GPU - quanto hanno allocato gli altri processi un istante fa.
     * Un controllo preventivo con cudaMemGetInfo confronterebbe una fotografia
     * gia' obsoleta e andrebbe corretto con un margine arbitrario; qui la
     * risposta la da' cudaMalloc, che non puo' sbagliarsi.
     *
     * La catena si ferma alla prima allocazione che non entra: le successive
     * non vengono nemmeno tentate, ed err arriva al controllo qui sotto. */
    /* Voce 2 del preprocessing: le allocazioni in VRAM. Crescono con la
     * taglia del blocco locale ma non sono un trasferimento. */
    const double t_device_alloc_start = now_seconds();
    err = cudaMalloc((void **)&local_gemm_context->dA_loc, nonzero(bytes_A));
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&local_gemm_context->dX_loc, nonzero(bytes_X));
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&local_gemm_context->dY_loc_part, nonzero(bytes_Y));

    if (err == cudaErrorMemoryAllocation) {
        /* Serve solo a comporre il messaggio: se anche questa query fallisse,
         * i due valori resterebbero a zero e la diagnosi degraderebbe senza
         * mascherare l'errore vero. Meglio questo del laconico "out of
         * memory" del runtime, che non dice ne' quanto serviva ne' che fare. */
        cudaMemGetInfo(&free_b, &total_b);
        die("%s: out of memory on device %d: the local block needs about "
            "%.2f GiB (A %dx%d, X %dx%d, Y %dx%d in %s), but only %.2f GiB "
            "of %.2f GiB were free: use more MPI processes or a smaller M/N",
            kernel_name(), CUDA_DEVICE_ID, (double)need / BYTES_PER_GIB,
            m_loc, lda, n_loc, ldx, m_loc, ldy, SCALAR_NAME,
            (double)free_b / BYTES_PER_GIB, (double)total_b / BYTES_PER_GIB);
    }
    CUDA_CHECK(err);
    local_gemm_context->t_setup_device_alloc = now_seconds() - t_device_alloc_start;

    /* A non cambia mai fra un'invocazione e l'altra: la H2D avviene una volta
     * sola, qui nel preprocessing. X e Y sono gia' dimensionate sopra, quindi
     * nessuna cudaMalloc puo' cadere nella prima repetition, neppure con
     * --warmup 0. */
    /* Voce 3 del preprocessing, e l'unica delle tre che e' PCIe: e' il
     * trasferimento che la separazione fra create e local_gemm ha tolto dal
     * cammino misurato, ed e' quindi quello che va discusso a parte. */
    const double t_h2d_A_start = now_seconds();
    if (bytes_A > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dA_loc, A_loc, bytes_A, cudaMemcpyHostToDevice));
    /* Una copia da memoria paginabile puo' ritornare prima che il DMA sia
     * concluso: senza questa attesa il tempo misurato sarebbe quello di
     * accodamento, non quello del trasferimento. */
    CUDA_CHECK(cudaDeviceSynchronize());
    local_gemm_context->t_setup_h2d_A = now_seconds() - t_h2d_A_start;

    /* Byte che attraversano davvero il bus, nelle stesse condizioni in cui
     * sono stati cronometrati: e' cio' che rende i tempi convertibili in
     * banda e confrontabili con il picco del PCIe. */
    local_gemm_context->bytes_h2d_A = (bytes_A > 0) ? bytes_A : 0;
    local_gemm_context->bytes_h2d_X_per_call =
        (n_loc > 0 && k > 0) ? (size_t)n_loc * (size_t)ldx * sizeof(scalar_t) : 0;
    local_gemm_context->bytes_d2h_Y_per_call =
        (m_loc > 0 && k > 0) ? (size_t)m_loc * (size_t)ldy * sizeof(scalar_t) : 0;

    CUDA_CHECK(cudaMemset(local_gemm_context->dY_loc_part, 0, nonzero(bytes_Y)));

    /* Gli event vanno creati una volta sola: crearli e distruggerli a ogni
     * invocazione sarebbe una chiamata al driver dentro la misura. */
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_kernel_start));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_kernel_stop));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_h2d_X_start));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_h2d_X_stop));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_d2h_Y_start));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_d2h_Y_stop));

    /* Le copie H2D da memoria paginabile possono ritornare prima che il DMA
     * sia terminato. Questa sincronizzazione garantisce che tutto il setup sia
     * davvero concluso prima di uscire dalla regione di preprocessing. */
    CUDA_CHECK(cudaDeviceSynchronize());

    local_gemm_context->t_setup = now_seconds() - t0;
    return local_gemm_context;
}

void local_gemm(local_gemm_t *local_gemm_context, const scalar_t *RESTRICT X_loc, int ldx, scalar_t *RESTRICT Y_loc_part, int ldy)
{
    const int m_loc = local_gemm_context->m_loc, n_loc = local_gemm_context->n_loc, k = local_gemm_context->k;
    const long long threads = (long long)m_loc * (long long)k;
    float ms = 0.0f;

    // Guardia: le leading dimension non possono cambiare fra le chiamate
    if (ldx != local_gemm_context->ldx || ldy != local_gemm_context->ldy)
        die("cuda_naive: leading dimensions changed between calls "
            "(ldx %d -> %d, ldy %d -> %d)",
            local_gemm_context->ldx, ldx, local_gemm_context->ldy, ldy);

    /* H2D di X_loc: una "fetta" n_loc x k, cioe' pochi MiB. E' dentro l'invocazione
     * perche' X_loc e' il risultato del MPI_Bcast appena concluso. */
    /* H2D di X: dentro l'invocazione perche' X e' il risultato del Bcast
     * appena concluso, ma delimitata dai suoi event per poterla sottrarre
     * dal tempo ufficiale e discuterla a parte. I record ci sono anche
     * quando non c'e' niente da copiare: cosi' il tempo e' definito
     * (circa zero) invece che indefinito. */
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_h2d_X_start, 0));
    if (n_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dX_loc, X_loc, (size_t)n_loc * (size_t)ldx * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_h2d_X_stop, 0));

    /* Gli event delimitano il SOLO kernel: quello che sta fra i due record e'
     * cio' che finisce in t_kernel e quindi in gflops_kernel. */
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_kernel_start, 0));
    if (threads > 0) {
        const int blocks = (int)((threads + BLOCK_THREADS - 1) / BLOCK_THREADS);
        naive_kernel<<<blocks, BLOCK_THREADS>>>(m_loc, n_loc, k, local_gemm_context->dA_loc, local_gemm_context->lda, local_gemm_context->dX_loc, ldx, local_gemm_context->dY_loc_part, ldy);
        /* Un lancio fallito non lancia eccezioni: si legge cosi'. */
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_kernel_stop, 0));

    /* D2H di Y.
     * Essendo sullo stream di default con memoria paginabile, questa
     * copia e' anche il punto di sincronizzazione: quando ritorna, il kernel ha
     * finito e Ypart e' pronta per la MPI_Reduce. */
    /* D2H di Y, l'altra copia esclusa dal tempo ufficiale. L'event di
     * partenza viene accodato dopo il kernel, quindi fra i due record
     * resta la sola copia e non l'attesa del calcolo. */
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_d2h_Y_start, 0));
    if (m_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(Y_loc_part, local_gemm_context->dY_loc_part, (size_t)m_loc * (size_t)ldy * sizeof(scalar_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_d2h_Y_stop, 0));

    /* Con blocco locale vuoto non c'e' stata nessuna copia a sincronizzare:
     * l'attesa esplicita sull'event serve comunque, ed e' gratuita quando il
     * lavoro e' gia' concluso. */
    /* L'ultimo event accodato e' quello della D2H: aspettare lui vuol dire
     * aspettare tutto il lavoro di questa invocazione. */
    CUDA_CHECK(cudaEventSynchronize(local_gemm_context->ev_d2h_Y_stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_kernel_start, local_gemm_context->ev_kernel_stop));
    local_gemm_context->t_last = (double)ms * 1.0e-3;

    /* Le due voci da discutere a parte. Insieme a t_last e a cio' che
     * avanza (overhead del runtime) ricostruiscono il tempo dell'intera
     * invocazione misurato dal chiamante. */
    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_h2d_X_start, local_gemm_context->ev_h2d_X_stop));
    local_gemm_context->t_last_h2d_X = (double)ms * 1.0e-3;

    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_d2h_Y_start, local_gemm_context->ev_d2h_Y_stop));
    local_gemm_context->t_last_d2h_Y = (double)ms * 1.0e-3;
}

void local_gemm_destroy(local_gemm_t *local_gemm_context)
{
    if (local_gemm_context == NULL)
        return;

    /* In chiusura gli errori non si possono piu' gestire in modo utile: si
     * libera quello che c'e' e si esce. */
    if (local_gemm_context->dA_loc != NULL) cudaFree(local_gemm_context->dA_loc);
    if (local_gemm_context->dX_loc != NULL) cudaFree(local_gemm_context->dX_loc);
    if (local_gemm_context->dY_loc_part != NULL) cudaFree(local_gemm_context->dY_loc_part);
    cudaEventDestroy(local_gemm_context->ev_kernel_start);
    cudaEventDestroy(local_gemm_context->ev_kernel_stop);
    cudaEventDestroy(local_gemm_context->ev_h2d_X_start);
    cudaEventDestroy(local_gemm_context->ev_h2d_X_stop);
    cudaEventDestroy(local_gemm_context->ev_d2h_Y_start);
    cudaEventDestroy(local_gemm_context->ev_d2h_Y_stop);
    xfree(local_gemm_context);
}

double local_gemm_last_compute_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_last : -1.0;
}

double local_gemm_setup_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_setup : 0.0;
}

/* ---------------------------------------------------------------------------
 * Tempi esclusi dalla misura ufficiale, misurati per essere discussi a parte
 * ------------------------------------------------------------------------- */
double local_gemm_last_h2d_X_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_last_h2d_X : -1.0;
}

double local_gemm_last_d2h_Y_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_last_d2h_Y : -1.0;
}

size_t local_gemm_bytes_h2d_A(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->bytes_h2d_A : 0;
}

size_t local_gemm_bytes_h2d_X_per_call(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->bytes_h2d_X_per_call : 0;
}

size_t local_gemm_bytes_d2h_Y_per_call(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->bytes_d2h_Y_per_call : 0;
}

double local_gemm_setup_device_init_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_setup_device_init : 0.0;
}

double local_gemm_setup_device_alloc_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_setup_device_alloc : 0.0;
}

double local_gemm_setup_h2d_A_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_setup_h2d_A : 0.0;
}

/* Questo backend non pianifica nessun tiling in shared memory: non ha il
 * concetto, e il sentinella negativo lo dice invece di inventare un numero. */
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

/* Il nome porta la dimensione del blocco quando non e' quella di default: nel
 * CSV le righe di uno sweep su BLOCK devono restare distinguibili fra loro. */
#define STR_(x) #x
#define STR(x)  STR_(x)

#if BLOCK_THREADS == 256
#define BLK_SUFFIX ""
#else
#define BLK_SUFFIX "(blk" STR(BLOCK_THREADS) ")"
#endif

const char *kernel_name(void)
{
    return "cuda_naive" BLK_SUFFIX;
}
