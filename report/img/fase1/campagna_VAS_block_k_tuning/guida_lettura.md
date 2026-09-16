Partiamo dalla prestazione normalizzata di ogni blocco per ciascun \(k\):

\[
R(B,k)=
\frac{P(B,k)}
{\max_{B'}P(B',k)},
\]

dove \(P(B,k)\) è `gflops_kernel`.

La perdita rispetto al blocco migliore è:

\[
L(B,k)=1-R(B,k).
\]

In percentuale:

\[
L_\%(B,k)=100-R_\%(B,k).
\]

### Perdita media

È la media delle perdite sui cinque valori di \(k\):

\[
L_{\mathrm{media}}(B)
=
\frac{1}{5}
\sum_{k\in\{3,6,8,20,32\}} L_\%(B,k).
\]

Equivalentemente:

\[
L_{\mathrm{media}}(B)
=
100-\overline{R_\%(B,k)}.
\]

Per `BLOCK=128`:

\[
L_{\mathrm{media}}=100-99{,}08=0{,}92\%.
\]

Significa che, mediamente sui cinque \(k\), il blocco 128 perde solamente lo 0,92% rispetto al miglior blocco specifico di ogni \(k\).

### Perdita massima

È la perdita più grande osservata tra i cinque valori:

\[
L_{\max}(B)
=
\max_k L_\%(B,k).
\]

Equivalentemente:

\[
L_{\max}(B)
=
100-\min_k R_\%(B,k).
\]

Per `BLOCK=128`, la prestazione normalizzata minima è 98,31%, ottenuta con \(k=8\). Quindi:

\[
L_{\max}=100-98{,}31=1{,}69\%.
\]

Nel grafico di destra:

- il cerchio rappresenta la **perdita media**;
- la croce rappresenta la **perdita massima**;
- il segmento mostra l’intervallo tra le due.

Un blocco è robusto quando entrambi i valori sono piccoli, soprattutto la perdita massima.