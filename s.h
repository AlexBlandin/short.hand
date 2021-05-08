/* (cosmo)short.hand */
#ifndef SHORTHAND_H_INCLUDED
#define SHORTHAND_H_INCLUDED

#ifndef COSMOPOLITAN_H_ /* proxy for "noncosmipolitan" setup */
#include <assert.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#else
/* trialing Cosmopolitan libc support, cosmopolitan.h has all the standard ones I use */
#include "cosmo/cosmopolitan.h" /* we assumed cosmipolitan above, this is just for linting etc. */
#endif

typedef unsigned int uint;

typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef uint8_t  u8;

#define u64_MAX UINT64_MAX
#define u32_MAX UINT32_MAX
#define u16_MAX UINT16_MAX
#define u8_MAX  UINT8_MAX

typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef int8_t  i8;

#define i64_MAX INT64_MAX
#define i32_MAX INT32_MAX
#define i16_MAX INT16_MAX
#define i8_MAX  INT8_MAX
#define i64_MIN INT64_MIN
#define i32_MIN INT32_MIN
#define i16_MIN INT16_MIN
#define i8_MIN  INT8_MIN

typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;
typedef int8_t  s8;

#define s64_MAX INT64_MAX
#define s32_MAX INT32_MAX
#define s16_MAX INT16_MAX
#define s8_MAX  INT8_MAX
#define s64_MIN INT64_MIN
#define s32_MIN INT32_MIN
#define s16_MIN INT16_MIN
#define s8_MIN  INT8_MIN

typedef double f64;
typedef float f32;

/* fallback bool defs */
#ifndef bool
  #ifdef _Bool
    #define bool _Bool
  #else
    #define bool char
  #endif
#endif

#ifndef true
  #define true 1
#endif
#ifndef false
  #define false 0
#endif

/* <iso646.h> for `and`, `or`, `not`, etc. */
#include <iso646.h>
/* fallback <iso646.h> */
#ifndef and
  #define and &&
#endif
#ifndef or
  #define or ||
#endif
#ifndef not
  #define not !
#endif

/* copy n bytes from src to dst */
#define copy(n_bytes, src, dst) memcpy(&dst, &src, n_bytes)

/* zero all bytes of source according to sizeof(src) */
#define zero(src) memset(&src, 0, sizeof(src))

#ifndef swap
#define swap(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))
#endif

#ifndef even
#define even(x) (!((x)&1))
#endif

#ifndef odd
#define odd(x) ((x)&1)
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef clamp
/* clamp the value of x to the range l..r */
#define clamp(x, l, r) ((x) < (l) ? (l) : (x) > (r) ? (r) : (x))
#endif

#ifndef lerp
#define lerp(a, b, t) ((a) + (t) * (float)((b) - (a)))
#endif

#ifndef unlerp
#define unlerp(a, b, t) (((t) - (a)) / (float)((b) - (a)))
#endif

#ifndef is_pow2
#define is_pow2(n) ((n) && !((n) & ((n)-1)))
#endif

#ifndef which_bit
/* given a value with only a single bit set, which bit is it? */
/* (https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog) */
u32 which_bit(u32 v) {
  static const u32 b[] = {0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000};
  // register u32 r = (v & b[0]) != 0;
  // for (u8 i = 4; i > 0; i--)
  //   r |= ((v & b[i]) != 0) << i;
  // return r;
  
  return ((v & b[0]) != 0)
       | ((v & b[1]) != 0) << 1
       | ((v & b[2]) != 0) << 2
       | ((v & b[3]) != 0) << 3
       | ((v & b[4]) != 0) << 4;
}
#endif

/* n bits to ceil(n/8) bytes */
#define bits(n) (((n) + 7) / 8)

/* x's value in the n'th bit */
#define bs_in(x, n) ((x)[(n) / 8] & (1 << ((n) % 8)))

/* set x's n'th bit to 1 */
#define bs_set(x, n) ((x)[(n) / 8] |= (1 << ((n) % 8)))

/* set x's n'th bit to 0 */
#define bs_clear(x, n) ((x)[(n) / 8] &= ~(1 << ((n) % 8)))

/* set x's bits from n on to 0 */
#define bs_clearall(x, n) ((x)[(n) / 8] &= ~((1 << (((n) % 8) + 1))-1))

/* toggle x's n'th bit */
#define bs_toggle(x, n) ((x)[(n) / 8] ^= (1 << ((n) % 8)))

#ifndef bitcount
i32 bitcount64(u64 v) { /* clang-10 favoured bc. 64 bit magic??? */
  i32 r = 0;
  while (v != 0) {
    v &= v - 1;
    r++;
  }
  return r;
}
#endif

/* from bit twiddling hacks (https://graphics.stanford.edu/~seander/bithacks.html) */
#ifndef bitcount32
i32 bitcount32(u32 v) { /* gcc favoured (clang converts to popcount, popcount is slower???) */
  v = v - ((v >> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
  return (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
}
#endif

#ifndef prime
/* quick primality test, inspired by stb.h (https://github.com/nothings/stb) */
bool prime(u32 n) {
  if (n == 2 || n == 3 || n == 5 || n == 7) return true;
  if (!(n & 1) || !(n % 3) || !(n % 5) || !(n % 7)) return false;
  if (n < 121) return n > 1;
  if (!(n % 121) || !(n % 123)) return false; // (127 is first prime after 113)
  for (u32 i = 125; i * i <= n; i += 6) // start after 121, from 11+6+6+6...
    if (!(n % i) || !(n % (i + 2))) return false;

  return true;
}
#endif

/* just prints a newline */
void println() {
  #ifdef COSMOPOLITAN_H_
    printf("%n");
  #else
    #ifdef WIN32
      printf("\r\n");
    #else
      printf("\n");
    #endif
  #endif
}

/* 2 u32s into a u64, `a` goes into "left" (higher) bits */
#define packu64(a, b) (((u64)(a)) << 32 | (b))

static u64 rng_state[4]; /* xoshiro256** state */
#define xs_rotl(x, k) ((x << k) | (x >> (64 - k)))
u64 randint(void) { /* xoshiro256** PRNG */
  const u64 result = xs_rotl(rng_state[1] * 5, 7) * 9, t = rng_state[1] << 17;

  rng_state[2] ^= rng_state[0];
  rng_state[3] ^= rng_state[1];
  rng_state[1] ^= rng_state[2];
  rng_state[0] ^= rng_state[3];

  rng_state[2] ^= t;
  rng_state[3] = xs_rotl(rng_state[3], 45);

  return result;
}

static bool _unseeded = true;
void seed_rng() {
  if (_unseeded) {
    srand((unsigned int)time(NULL));
    for (int i = 4; i--;) rng_state[i] = packu64(rand(), rand());
  }
  _unseeded = false;
}

/* Justine Tunney here to save the day https://justine.lol/endian.html */
#define READ16LE(P) ((255 & (P)[1]) << 8 | (255 & (P)[0]))
#define READ16BE(P) ((255 & (P)[0]) << 8 | (255 & (P)[1]))
#define READ32LE(P)                                                    \
  ((uint32_t)(255 & (P)[3]) << 030 | (uint32_t)(255 & (P)[2]) << 020 | \
   (uint32_t)(255 & (P)[1]) << 010 | (uint32_t)(255 & (P)[0]) << 000)
#define READ32BE(P)                                                    \
  ((uint32_t)(255 & (P)[0]) << 030 | (uint32_t)(255 & (P)[1]) << 020 | \
   (uint32_t)(255 & (P)[2]) << 010 | (uint32_t)(255 & (P)[3]) << 000)
#define READ64LE(P)                                                    \
  ((uint64_t)(255 & (P)[7]) << 070 | (uint64_t)(255 & (P)[6]) << 060 | \
   (uint64_t)(255 & (P)[5]) << 050 | (uint64_t)(255 & (P)[4]) << 040 | \
   (uint64_t)(255 & (P)[3]) << 030 | (uint64_t)(255 & (P)[2]) << 020 | \
   (uint64_t)(255 & (P)[1]) << 010 | (uint64_t)(255 & (P)[0]) << 000)
#define READ64BE(P)                                                    \
  ((uint64_t)(255 & (P)[0]) << 070 | (uint64_t)(255 & (P)[1]) << 060 | \
   (uint64_t)(255 & (P)[2]) << 050 | (uint64_t)(255 & (P)[3]) << 040 | \
   (uint64_t)(255 & (P)[4]) << 030 | (uint64_t)(255 & (P)[5]) << 020 | \
   (uint64_t)(255 & (P)[6]) << 010 | (uint64_t)(255 & (P)[7]) << 000)

#define WRITE16LE(P, V)                        \
  ((P)[0] = (0x00000000000000FF & (V)) >> 000, \
   (P)[1] = (0x000000000000FF00 & (V)) >> 010, (P) + 2)
#define WRITE16BE(P, V)                        \
  ((P)[0] = (0x000000000000FF00 & (V)) >> 010, \
   (P)[1] = (0x00000000000000FF & (V)) >> 000, (P) + 2)
#define WRITE32LE(P, V)                        \
  ((P)[0] = (0x00000000000000FF & (V)) >> 000, \
   (P)[1] = (0x000000000000FF00 & (V)) >> 010, \
   (P)[2] = (0x0000000000FF0000 & (V)) >> 020, \
   (P)[3] = (0x00000000FF000000 & (V)) >> 030, (P) + 4)
#define WRITE32BE(P, V)                        \
  ((P)[0] = (0x00000000FF000000 & (V)) >> 030, \
   (P)[1] = (0x0000000000FF0000 & (V)) >> 020, \
   (P)[2] = (0x000000000000FF00 & (V)) >> 010, \
   (P)[3] = (0x00000000000000FF & (V)) >> 000, (P) + 4)
#define WRITE64LE(P, V)                        \
  ((P)[0] = (0x00000000000000FF & (V)) >> 000, \
   (P)[1] = (0x000000000000FF00 & (V)) >> 010, \
   (P)[2] = (0x0000000000FF0000 & (V)) >> 020, \
   (P)[3] = (0x00000000FF000000 & (V)) >> 030, \
   (P)[4] = (0x000000FF00000000 & (V)) >> 040, \
   (P)[5] = (0x0000FF0000000000 & (V)) >> 050, \
   (P)[6] = (0x00FF000000000000 & (V)) >> 060, \
   (P)[7] = (0xFF00000000000000 & (V)) >> 070, (P) + 8)
#define WRITE64BE(P, V)                        \
  ((P)[0] = (0xFF00000000000000 & (V)) >> 070, \
   (P)[1] = (0x00FF000000000000 & (V)) >> 060, \
   (P)[2] = (0x0000FF0000000000 & (V)) >> 050, \
   (P)[3] = (0x000000FF00000000 & (V)) >> 040, \
   (P)[4] = (0x00000000FF000000 & (V)) >> 030, \
   (P)[5] = (0x0000000000FF0000 & (V)) >> 020, \
   (P)[6] = (0x000000000000FF00 & (V)) >> 010, \
   (P)[7] = (0x00000000000000FF & (V)) >> 000, (P) + 8)




#endif /* SHORTHAND_H_INCLUDED */
