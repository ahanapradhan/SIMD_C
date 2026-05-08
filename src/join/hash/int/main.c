/* 100% GPT 5.5 Generated Code, but the code structure and bug/perf fixes were instructed by me */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
#include <windows.h>


#define CACHE_FLUSH_SIZE (64 * 1024 * 1024)  // 64 MB (safe)

char *flush_buffer;

void init_flush_buffer() {
    flush_buffer = (char*) malloc(CACHE_FLUSH_SIZE);
    for (size_t i = 0; i < CACHE_FLUSH_SIZE; i++) {
        flush_buffer[i] = i;
    }
}

void flush_cache() {
    for (size_t i = 0; i < CACHE_FLUSH_SIZE; i += 64) {
        flush_buffer[i]++;
    }
}


typedef struct {
    int key;
    int value;
} KV;

#define EMPTY_KEY -1

/* ---------------- Timer ---------------- */
double now() {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / f.QuadPart;
}

/* ---------------- Next power of 2 ---------------- */
int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

/* ---------------- Hash ---------------- */
static inline int hash(int k, int mask) {
    return (k * 2654435761u) & mask;
}

/* ---------------- Build table ---------------- */
KV* build_table(KV* R, int n, int* out_size) {
    int size = next_pow2(n * 2);   // low load factor
    int mask = size - 1;

    KV* table = (KV*)malloc(sizeof(KV) * size);

    for (int i = 0; i < size; i++)
        table[i].key = EMPTY_KEY;

    for (int i = 0; i < n; i++) {
        int h = hash(R[i].key, mask);

        while (table[h].key != EMPTY_KEY) {
            h = (h + 1) & mask;
        }
        table[h] = R[i];
    }

    *out_size = size;
    return table;
}

/* ---------------- Baseline ---------------- */
long long join_baseline(KV* R, KV* S, int n) {
    int size;
    KV* table = build_table(R, n, &size);
    int mask = size - 1;

    long long matches = 0;

    for (int i = 0; i < n; i++) {
        int h = hash(S[i].key, mask);

        while (1) {
            if (table[h].key == EMPTY_KEY)
                break;

            if (table[h].key == S[i].key) {
                matches++;
                break;
            }
            h = (h + 1) & mask;
        }
    }

    free(table);
    return matches;
}

/* ---------------- TRUE SIMD VERSION ---------------- */

long long join_simd(KV* R, KV* S, int n) {
    int size;
    KV* table = build_table(R, n, &size);
    int mask = size - 1;

    // ✅ extract keys into flat array (critical fix)
    int* keys = (int*)malloc(sizeof(int) * (size + 8));

    for (int i = 0; i < size; i++)
        keys[i] = table[i].key;

    // padding
    for (int i = size; i < size + 8; i++)
        keys[i] = EMPTY_KEY;

    long long matches = 0;

    for (int i = 0; i < n; i++) {

        int key = S[i].key;

        __m256i key_vec   = _mm256_set1_epi32(key);
        __m256i empty_vec = _mm256_set1_epi32(EMPTY_KEY);

        int h = hash(key, mask);

        while (1) {

            // ✅ SAFE + CORRECT load (8 contiguous ints)
            __m256i vec =
                _mm256_loadu_si256((__m256i*)&keys[h]);

            __m256i cmp = _mm256_cmpeq_epi32(key_vec, vec);
            int match_mask = _mm256_movemask_epi8(cmp);

            if (match_mask) {
                matches++;
                break;
            }

            __m256i empty_cmp =
                _mm256_cmpeq_epi32(vec, empty_vec);

            int empty_mask = _mm256_movemask_epi8(empty_cmp);

            if (empty_mask) {
                break;
            }

            h = (h + 8) & mask;
        }
    }

    free(keys);
    free(table);

    return matches;
}
/* ---------------- Data ---------------- */
void gen(KV* R, KV* S, int n) {
    for (int i = 0; i < n; i++) {
        R[i].key = i;
        R[i].value = i;

        S[i].key = i;
        S[i].value = i;
    }
}

/* ---------------- Main ---------------- */
int main() {

    long sizes[] = {
        10000,
        50000,
        100000,
        500000,
        1000000,
        5000000,
        10000000,
        20000000,
        50000000,
        100000000,
        200000000,
        500000000
    };

    int tests = sizeof(sizes)/sizeof(sizes[0]);

    printf("Size,SIMD,Baseline\n");

    for (int t = 0; t < tests; t++) {

        int n = sizes[t];

        KV* R = (KV*)malloc(sizeof(KV)*n);
        KV* S = (KV*)malloc(sizeof(KV)*n);

        gen(R, S, n);

        init_flush_buffer();
        flush_cache();
        double t3 = now();
        long long m2 = join_simd(R, S, n);
        double t4 = now();

        init_flush_buffer();
        flush_cache();
        double t1 = now();
        long long m1 = join_baseline(R, S, n);
        double t2 = now();

        printf("%d,%.6f,%.6f (%d)\n",
               n, t2-t1, t4-t3, m1==m2);

        free(R);
        free(S);
    }

    return 0;
}
