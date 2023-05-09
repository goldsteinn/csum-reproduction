#include <assert.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <x86intrin.h>
#define __u32   uint32_t
#define u32     uint32_t
#define u64     uint64_t
#define __wsum  uint64_t
#define __sum16 uint16_t
#define __force

#define likely(x)   __builtin_expect(x, 1)
#define unlikely(x) __builtin_expect(x, 0)

static inline __u32
ror32(__u32 word, unsigned int shift) {
    return (word >> (shift & 31)) | (word << ((-shift) & 31));
}


static inline unsigned
add32_with_carry(unsigned a, unsigned b) {
    asm("addl %2,%0\n\t"
        "adcl $0,%0"
        : "=r"(a)
        : "0"(a), "rm"(b));
    return a;
}
static inline unsigned short
from32to16(unsigned a) {
    unsigned short b = a >> 16;
    asm("addw %w2,%w0\n\t"
        "adcw $0,%w0\n"
        : "=r"(b)
        : "0"(b), "r"(a));
    return b;
}

static inline unsigned long
load_unaligned_zeropad(const void * addr) {
    unsigned long ret;

    asm volatile(
        "1:	mov %[mem], %[ret]\n"
        "2:\n"
        : [ret] "=r"(ret)
        : [mem] "m"(*(unsigned long *)addr));

    return ret;
}


static inline __wsum
csum_tail(unsigned result, u64 temp64, int odd) {
    result = add32_with_carry(temp64 >> 32, temp64 & 0xffffffff);
    if (unlikely(odd)) {
        result = from32to16(result);
        result = ((result >> 8) & 0xff) | ((result & 0xff) << 8);
    }
    return (__force __wsum)result;
}

/*
 * Do a checksum on an arbitrary memory area.
 * Returns a 32bit checksum.
 *
 * This isn't as time critical as it used to be because many NICs
 * do hardware checksumming these days.
 *
 * Still, with CHECKSUM_COMPLETE this is called to compute
 * checksums on IPv6 headers (40 bytes) and other small parts.
 * it's best to have buff aligned on a 64-bit boundary
 */
__wsum __attribute__((aligned(4096), noinline, noclone))
csum_partial_new(const void * buff, int len, __wsum sum) {
    u64      temp64 = (__force u64)sum;
    unsigned odd, result;

    odd = 1 & (unsigned long)buff;
    if (unlikely(odd)) {
        if (unlikely(len == 0)) return sum;
        temp64 = ror32((__force u32)sum, 8);
        temp64 += (*(unsigned char *)buff << 8);
        len--;
        buff++;
    }

    /*
     * len == 40 is the hot case due to IPv6 headers, but annotating it likely()
     * has noticeable negative affect on codegen for all other cases with
     * minimal performance benefit here.
     */
    if (len == 40) {
        asm("addq 0*8(%[src]),%[res]\n\t"
            "adcq 1*8(%[src]),%[res]\n\t"
            "adcq 2*8(%[src]),%[res]\n\t"
            "adcq 3*8(%[src]),%[res]\n\t"
            "adcq 4*8(%[src]),%[res]\n\t"
            "adcq $0,%[res]"
            : [res] "+r"(temp64)
            : [src] "r"(buff), "m"(*(const char(*)[40])buff));
        return csum_tail(result, temp64, odd);
    }
#if 1
    if (unlikely(len >= 64)) {
        /*
         * Extra accumulators for better ILP in the loop.
         */
        u64 tmp_accum, tmp_carries;

        asm("xorl %k[tmp_accum],%k[tmp_accum]\n\t"
            "xorl %k[tmp_carries],%k[tmp_carries]\n\t"
            "subl $64, %[len]\n\t"
            "1:\n\t"
            "addq 0*8(%[src]),%[res]\n\t"
            "adcq 1*8(%[src]),%[res]\n\t"
            "adcq 2*8(%[src]),%[res]\n\t"
            "adcq 3*8(%[src]),%[res]\n\t"
            "adcl $0,%k[tmp_carries]\n\t"
            "addq 4*8(%[src]),%[tmp_accum]\n\t"
            "adcq 5*8(%[src]),%[tmp_accum]\n\t"
            "adcq 6*8(%[src]),%[tmp_accum]\n\t"
            "adcq 7*8(%[src]),%[tmp_accum]\n\t"
            "adcl $0,%k[tmp_carries]\n\t"
            "addq $64, %[src]\n\t"
            "subl $64, %[len]\n\t"
            "jge 1b\n\t"
            "addq %[tmp_accum],%[res]\n\t"
            "adcq %[tmp_carries],%[res]\n\t"
            "adcq $0,%[res]"
            : [tmp_accum] "=&r"(tmp_accum), [tmp_carries] "=&r"(tmp_carries),
              [res] "+r"(temp64), [len] "+r"(len), [src] "+r"(buff)
            : "m"(*(const char *)buff));
    }
#else
    if (unlikely(len >= 64)) {
        int chunks = len >> 6;
        asm("clc\n\t"
            "1:\n\t"
            "adcq 0*8(%[src]),%[res]\n\t"
            "adcq 1*8(%[src]),%[res]\n\t"
            "adcq 2*8(%[src]),%[res]\n\t"
            "adcq 3*8(%[src]),%[res]\n\t"
            "adcq 4*8(%[src]),%[res]\n\t"
            "adcq 5*8(%[src]),%[res]\n\t"
            "adcq 6*8(%[src]),%[res]\n\t"
            "adcq 7*8(%[src]),%[res]\n\t"
            /* Leave carry flag unchanged.  */
            "leaq 64(%[src]), %[src]\n\t"
            "decl %[chunks]\n\t"
            "jnz 1b\n\t"
            "adcq $0,%[res]"
            : [res] "+r"(temp64), [chunks] "+r"(chunks), [src] "+r"(buff)
            : "m"(*(const char *)buff));
    }
#endif

    if (len & 32) {
        asm("addq 0*8(%[src]),%[res]\n\t"
            "adcq 1*8(%[src]),%[res]\n\t"
            "adcq 2*8(%[src]),%[res]\n\t"
            "adcq 3*8(%[src]),%[res]\n\t"
            "adcq $0,%[res]"
            : [res] "+r"(temp64)
            : [src] "r"(buff), "m"(*(const char(*)[32])buff));
        buff += 32;
    }
    if (len & 16) {
        asm("addq 0*8(%[src]),%[res]\n\t"
            "adcq 1*8(%[src]),%[res]\n\t"
            "adcq $0,%[res]"
            : [res] "+r"(temp64)
            : [src] "r"(buff), "m"(*(const char(*)[16])buff));
        buff += 16;
    }
    if (len & 8) {
        asm("addq 0*8(%[src]),%[res]\n\t"
            "adcq $0,%[res]"
            : [res] "+r"(temp64)
            : [src] "r"(buff), "m"(*(const char(*)[8])buff));
        buff += 8;
    }
    if (len & 7) {
        unsigned int  shift = (8 - (len & 7)) * 8;
        unsigned long trail;

        trail = (load_unaligned_zeropad(buff) << shift) >> shift;

        asm("addq %[trail],%[res]\n\t"
            "adcq $0,%[res]"
            : [res] "+r"(temp64)
            : [trail] "r"(trail));
    }
    return csum_tail(result, temp64, odd);
}
__wsum __attribute__((aligned(4096), noinline, noclone))
csum_partial_old(const void * buff, int len, __wsum sum) {
    u64      temp64 = (__force u64)sum;
    unsigned odd, result;

    odd = 1 & (unsigned long)buff;
    if (unlikely(odd)) {
        if (unlikely(len == 0)) return sum;
        temp64 = ror32((__force u32)sum, 8);
        temp64 += (*(unsigned char *)buff << 8);
        len--;
        buff++;
    }

    while (unlikely(len >= 64)) {
        asm("addq 0*8(%[src]),%[res]\n\t"
            "adcq 1*8(%[src]),%[res]\n\t"
            "adcq 2*8(%[src]),%[res]\n\t"
            "adcq 3*8(%[src]),%[res]\n\t"
            "adcq 4*8(%[src]),%[res]\n\t"
            "adcq 5*8(%[src]),%[res]\n\t"
            "adcq 6*8(%[src]),%[res]\n\t"
            "adcq 7*8(%[src]),%[res]\n\t"
            "adcq $0,%[res]"
            : [res] "+r"(temp64)
            : [src] "r"(buff)
            : "memory");
        buff += 64;
        len -= 64;
    }

    if (len & 32) {
        asm("addq 0*8(%[src]),%[res]\n\t"
            "adcq 1*8(%[src]),%[res]\n\t"
            "adcq 2*8(%[src]),%[res]\n\t"
            "adcq 3*8(%[src]),%[res]\n\t"
            "adcq $0,%[res]"
            : [res] "+r"(temp64)
            : [src] "r"(buff)
            : "memory");
        buff += 32;
    }
    if (len & 16) {
        asm("addq 0*8(%[src]),%[res]\n\t"
            "adcq 1*8(%[src]),%[res]\n\t"
            "adcq $0,%[res]"
            : [res] "+r"(temp64)
            : [src] "r"(buff)
            : "memory");
        buff += 16;
    }
    if (len & 8) {
        asm("addq 0*8(%[src]),%[res]\n\t"
            "adcq $0,%[res]"
            : [res] "+r"(temp64)
            : [src] "r"(buff)
            : "memory");
        buff += 8;
    }
    if (len & 7) {
        unsigned int  shift = (8 - (len & 7)) * 8;
        unsigned long trail;

        trail = (load_unaligned_zeropad(buff) << shift) >> shift;

        asm("addq %[trail],%[res]\n\t"
            "adcq $0,%[res]"
            : [res] "+r"(temp64)
            : [trail] "r"(trail));
    }
    result = add32_with_carry(temp64 >> 32, temp64 & 0xffffffff);
    if (unlikely(odd)) {
        result = from32to16(result);
        result = ((result >> 8) & 0xff) | ((result & 0xff) << 8);
    }
    return (__force __wsum)result;
}


typedef struct rand_vals {
    uint32_t pos_;
    uint64_t rvals_[512];
} rand_vals_t;

uint64_t
get_rval(rand_vals_t * rvals) {
    if (rvals->pos_ == 64) {
        assert(getrandom(rvals->rvals_, 512 * 8, 0) == 512 * 8);
        rvals->pos_ = 0;
    }

    return rvals->rvals_[rvals->pos_++];
}

static void
tester() {
    rand_vals_t rvals;
    int         rerun = 0;
    rvals.pos_        = 0;


    enum { PAGE_SIZE = 0, PAGE_END = 4096 - 8, MAX_LEN = 4096 + PAGE_END };

    /* Linux overreads small loads and just handles pagefaults internally. Hard
     * to emulate in userland so allocate extra 8-bytes.  */
    uint8_t * buf = malloc(MAX_LEN + 8);

    for (;;) {
        fprintf(stderr, "%sunning Random Input Exhaustive Size/Align Tests",
                rerun ? "Rer" : "R");
        assert(getrandom(buf, MAX_LEN, 0) == MAX_LEN);
        for (uint64_t align = 0; align < MAX_LEN; ++align) {
            for (uint64_t len = 0; len < MAX_LEN; ++len) {
                if (len + align >= MAX_LEN) {
                    continue;
                }
                uint64_t wsum = get_rval(&rvals);
                assert(csum_partial_old(buf, len, wsum) ==
                       csum_partial_new(buf, len, wsum));
            }
        }
        fprintf(stderr, " -- Success\n");
        rerun = 1;
    }
}

#define BENCH_KERNEL_LAT(func, len)                                            \
 ({                                                                            \
uint32_t tmp_trials_ = TRIALS;                                                  \
uint64_t tmp_sum_    = 0;                                                       \
uint64_t start, end;                                                            \
start = _rdtsc();                                                               \
__asm__ volatile(".p2align 6\n" : : :);                                         \
for (; tmp_trials_; --tmp_trials_) {                                            \
tmp_sum_ = func(buf, len, tmp_sum_);                                           \
}                                                                               \
__asm__ volatile("" : : "g"(tmp_sum_) : "memory");                              \
end = _rdtsc();                                                                 \
                                                                               \
end - start;                                                                    \
 });

#define BENCH_KERNEL_TPUT(func, len)                                           \
 ({                                                                            \
uint32_t tmp_trials_ = TRIALS;                                                  \
uint64_t tmp_sum_    = 0;                                                       \
uint64_t start, end;                                                            \
start = _rdtsc();                                                               \
__asm__ volatile(".p2align 6\n" : : :);                                         \
for (; tmp_trials_; --tmp_trials_) {                                            \
__asm__ volatile("" ::"g"(func(buf, len, tmp_sum_)) : "memory");               \
}                                                                               \
                                                                               \
end = _rdtsc();                                                                 \
                                                                               \
end - start;                                                                    \
 });

#define BENCH_CONF(len)                                                        \
 {                                                                             \
  uint64_t tmp_lat_new_  = BENCH_KERNEL_LAT(csum_partial_new, len);            \
  uint64_t tmp_tput_new_ = BENCH_KERNEL_TPUT(csum_partial_new, len);           \
  uint64_t tmp_lat_old_  = BENCH_KERNEL_LAT(csum_partial_old, len);            \
  uint64_t tmp_tput_old_ = BENCH_KERNEL_TPUT(csum_partial_old, len);           \
  *(confs) = (bench_conf_t){ len, tmp_lat_new_, tmp_tput_new_, tmp_lat_old_,   \
                             tmp_tput_old_ };                                  \
  ++confs;                                                                     \
 }

typedef struct bench_conf {
    uint32_t len_;
    uint64_t lat_new_;
    uint64_t tput_new_;
    uint64_t lat_old_;
    uint64_t tput_old_;
} bench_conf_t;
#define TT(val)  (((double)(val)) / ((double)TRIALS))
#define TT2(val) (((double)(val)) / (((double)TRIALS) * ((double)(cnt - 4))))

#define ALL_RUNS                                                               \
 BENCH_CONF(8);                                                                \
 BENCH_CONF(16);                                                               \
 BENCH_CONF(24);                                                               \
 BENCH_CONF(32);                                                               \
 BENCH_CONF(40);                                                               \
 BENCH_CONF(48);                                                               \
 BENCH_CONF(56);                                                               \
 BENCH_CONF(64);                                                               \
 BENCH_CONF(96);                                                               \
 BENCH_CONF(128);                                                              \
 BENCH_CONF(200);                                                              \
 BENCH_CONF(272);                                                              \
 BENCH_CONF(256 + 128 + 32 + 16 + 8);                                          \
 BENCH_CONF(512 + 256 + 128 + 32 + 16 + 8);                                    \
 BENCH_CONF(1024);                                                             \
 BENCH_CONF(1024 + 512 + 16);                                                  \
 BENCH_CONF(2048);                                                             \
 BENCH_CONF(2048 + 512 + 32 + 8);                                              \
 BENCH_CONF(2048 + 1024 + 512 + 16 + 8);                                       \
 BENCH_CONF(4096);


static void __attribute__((aligned(4096), noinline, noclone)) bench() {
    enum { TRIALS = 1000 * 1000 };
    uint8_t * buf = aligned_alloc(4096, 4096);
    memset(buf, 0, 4096);

    bench_conf_t  confs_accum[64];
    bench_conf_t  confs_base[64];
    bench_conf_t *confs, *confs_start, *accum;
    uint32_t      cnt = 0;
    memset(confs_accum, 0, sizeof(confs_accum));

    confs = &confs_base[0];
    ALL_RUNS;
    for (;;) {
        confs = &confs_base[0];
        ALL_RUNS;
        ++cnt;
        if (cnt > 4) {
            confs_start = &confs_base[0];
            accum       = &confs_accum[0];
            for (; confs_start != confs; ++confs_start) {
                accum->len_ = confs_start->len_;
                accum->lat_new_ += confs_start->lat_new_;
                accum->lat_old_ += confs_start->lat_old_;
                accum->tput_new_ += confs_start->tput_new_;
                accum->tput_old_ += confs_start->tput_old_;
#if 0
            fprintf(
                stderr,
                "%4u  %8.2lf  %8.2lf  %5.3lf    %8.2lf  %8.2lf  %5.3lf\n",
                confs_start->len_, TT(confs_start->lat_new_),
                TT(confs_start->lat_old_),
                TT(confs_start->lat_new_) / TT(confs_start->lat_old_),
                TT(confs_start->tput_new_), TT(confs_start->tput_old_),
                TT(confs_start->tput_new_) / TT(confs_start->tput_old_));
#endif
                ++accum;
            }


            if ((cnt % 4) == 0) {
                fprintf(
                    stderr,
                    "-----------------------------------------------------------------\n");
                fprintf(stderr, "%4s  %8s  %8s  %5s    %8s  %8s  %5s\n", "len",
                        "lat_new", "lat_old", "r", "tput_new", "tput_old", "r");

                confs_start = &confs_accum[0];
                for (; confs_start != accum; ++confs_start) {
                    fprintf(
                        stderr,
                        "%4u  %8.2lf  %8.2lf  %5.3lf    %8.2lf  %8.2lf  %5.3lf\n",
                        confs_start->len_, TT2(confs_start->lat_new_),
                        TT2(confs_start->lat_old_),
                        TT2(confs_start->lat_new_) / TT2(confs_start->lat_old_),
                        TT2(confs_start->tput_new_),
                        TT2(confs_start->tput_old_),
                        TT2(confs_start->tput_new_) /
                            TT2(confs_start->tput_old_));
                }
                fprintf(
                    stderr,
                    "-----------------------------------------------------------------\n");
            }
        }
    }
}

int
main(int argc, char ** argv) {
    if (argc > 1 && argv[1][0] == 't') {
        tester();
    }
    else if (argc > 1 && argv[1][0] == 'b') {
        printf("Running Benchmarks. This may take a moment to display\n");
        bench();
    }
    printf("Usage: %s <'t' or 'b' for tests/bench respective>\n", argv[0]);
}
