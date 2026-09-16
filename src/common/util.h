#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

void *xmalloc(size_t bytes);
void  xfree(void *p);

double now_seconds(void);

#if defined(__GNUC__)
#define PRINTF_FMT(fmt_idx, arg_idx) \
    __attribute__((format(printf, fmt_idx, arg_idx)))
#else
#define PRINTF_FMT(fmt_idx, arg_idx)
#endif

void die(const char *fmt, ...) PRINTF_FMT(1, 2);

#if defined(__cplusplus)
}
#endif

#endif
