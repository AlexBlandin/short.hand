/* short.hand */
#ifndef SHORTHAND_H_INCLUDED
#define SHORTHAND_H_INCLUDED

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef uint8_t u8;

#define u64_MAX UINT64_MAX
#define u32_MAX UINT32_MAX
#define u16_MAX UINT16_MAX
#define u8_MAX UINT8_MAX

typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef int8_t i8;

#define i64_MAX INT64_MAX
#define i32_MAX INT32_MAX
#define i16_MAX INT16_MAX
#define i8_MAX INT8_MAX
#define i64_MIN INT64_MIN
#define i32_MIN INT32_MIN
#define i16_MIN INT16_MIN
#define i8_MIN INT8_MIN

typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;
typedef int8_t s8;

#define s64_MAX INT64_MAX
#define s32_MAX INT32_MAX
#define s16_MAX INT16_MAX
#define s8_MAX INT8_MAX
#define s64_MIN INT64_MIN
#define s32_MIN INT32_MIN
#define s16_MIN INT16_MIN
#define s8_MIN INT8_MIN

typedef double f64;
typedef float f32;

#ifdef _WIN32
/* use in a "... %"LU" ..." pattern, allows formatting */
#define LU "llu"
#define LD "lld"
#define LN "\r\n"
#else
/* use in a "... %"LU" ..." pattern, allows formatting */
#define LU "lu"
#define LD "ld"
#define LN "\n"
#endif

#ifndef printvar
#define printvar(var) printf("%s=%d"LN,(#var),(var))
#endif

/* just prints a newline */
#define println() printf(LN)

#if !defined(bool) && defined(_Bool)
#define bool _Bool
#endif

#if !defined(bool) && !defined(_Bool)
#define bool char
#endif

#define not !
#define and &&
#define or ||

#ifndef true
#define true 1
#endif
#ifndef false
#define false 0
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

#ifndef PI
#define PI 3.141592653589793238
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
#define bs_in(x, n) ((x)[(n)/8] & (1 << ((n) % 8)))

/* set x's n'th bit to 1 */
#define bs_set(x, n) ((x)[(n)/8] |= (1 << ((n) % 8)))

/* set x's n'th bit to 0 */
#define bs_clear(x, n) ((x)[(n)/8] &= ~(1 << ((n) % 8)))

/* set x's bits from n on to 0 */
#define bs_clearall(x, n) ((x)[(n)/8] &= ~((1 << (((n) % 8) + 1))-1))

/* toggle x's n'th bit */
#define bs_toggle(x, n) ((x)[(n)/8] ^= (1 << ((n) % 8)))

#if !defined(bitcount64)
i32 bitcount64(uint64_t v) { /* clang-10 favoured bc. 64 bit magic??? */
  i32 r = 0;
  while(v != 0) {
    v &= v - 1;
    r++;
  }
  return r;
}
#endif

/* from bit twiddling hacks (https://graphics.stanford.edu/~seander/bithacks.html) */
#if !defined(bitcount32)
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

  for (u32 i = 11; i * i <= n; i += 6)
    if (!(n % i) || !(n % (i + 2))) return false;

  return true;
}
#endif

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

#endif /* SHORTHAND_H_INCLUDED */
