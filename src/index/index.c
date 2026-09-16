#include "index/index.h"

#include <assert.h>

int block_size(int n, int p, int i)
{
    assert(p > 0 && i >= 0 && i < p && n >= 0);
    return n / p + (i < n % p ? 1 : 0);
}

int block_start(int n, int p, int i)
{
    int q, r;
    assert(p > 0 && i >= 0 && i <= p && n >= 0);
    q = n / p;
    r = n % p;

    return i * q + (i < r ? i : r);
}

int block_owner(int n, int p, int g)
{
    int q, r, cut;
    assert(p > 0 && g >= 0 && g < n);
    q = n / p;
    r = n % p;
    cut = r * (q + 1);
    if (g < cut)
        return g / (q + 1);
    return r + (g - cut) / q;
}
