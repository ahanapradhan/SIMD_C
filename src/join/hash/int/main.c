#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
#include <windows.h>

typedef struct {
    int key;
    int value;
} KV;

#define EMPTY_KEY -1
#define SIMD_WIDTH 8

/*
   Tune this.

   8 bits = 256 partitions.
   If radix is still too expensive, try 6 or 7.

   Recommended tests:
       RADIX_BITS = 6
       RADIX_BITS = 7
       RADIX_BITS = 8
*/
#define RADIX_BITS 8
#define NUM_PARTITIONS (1 << RADIX_BITS)

/* ---------------- Timer ---------------- */

double now(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / f.QuadPart;
}

/* ---------------- Cache flush ---------------- */

#define CACHE_FLUSH_SIZE (64 * 1024 * 1024)

char *flush_buffer = NULL;

void init_flush_buffer(void)
{
    flush_buffer = (char*)malloc(CACHE_FLUSH_SIZE);

    if (flush_buffer == NULL) {
        printf("Failed to allocate cache flush buffer\n");
        exit(1);
    }

    for (size_t i = 0; i < CACHE_FLUSH_SIZE; i++) {
        flush_buffer[i] = (char)i;
    }
}

void flush_cache(void)
{
    if (flush_buffer == NULL) {
        return;
    }

    for (size_t i = 0; i < CACHE_FLUSH_SIZE; i += 64) {
        flush_buffer[i]++;
    }
}

/* ---------------- Utility ---------------- */

int next_pow2_int(int x)
{
    int p = 1;

    while (p < x) {
        p <<= 1;
    }

    return p;
}

/*
   Hash function A.

   Used ONLY for radix partitioning.
   We use the HIGH bits of this hash for partition ID.
*/
static inline uint32_t hash32_partition(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/*
   Hash function B.

   Used ONLY for local hash tables and normal hash join.

   This is deliberately different from hash32_partition().
   This prevents the earlier bug where partitioning and local placement
   reused the same low hash bits.
*/
static inline uint32_t hash32_table(uint32_t x)
{
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;
    return x;
}

/*
   Partition ID uses HIGH bits.

   This is important:
   if partitioning uses low bits and local hashing also uses low bits,
   then each partition has fixed low bits and local hash tables cluster badly.
*/
static inline int get_partition_id(int key)
{
    uint32_t h = hash32_partition((uint32_t)key);
    return (int)(h >> (32 - RADIX_BITS));
}

/*
   Local hash uses a different hash function.
*/
static inline int local_hash(int key, int mask)
{
    uint32_t h = hash32_table((uint32_t)key);
    return (int)(h & (uint32_t)mask);
}

/* ---------------- Data generation ---------------- */

/*
   Guaranteed unique positive keys for n <= 2^31.

   The permutation is deterministic.
   S is shuffled so probing order is random.
*/

static inline int make_unique_key_strict(int i)
{
    uint32_t x = (uint32_t)i;

    x = (uint32_t)(((uint64_t)x * 1103515245u + 12345u) & 0x7FFFFFFFu);

    return (int)x;
}

static inline uint32_t fast_rand32(uint32_t *state)
{
    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    *state = x;
    return x;
}

void gen(KV* R, KV* S, int n)
{
    for (int i = 0; i < n; i++) {
        int key = make_unique_key_strict(i);

        R[i].key = key;
        R[i].value = i;

        S[i].key = key;
        S[i].value = i;
    }

    uint32_t seed = 123456789u;

    for (int i = n - 1; i > 0; i--) {
        uint32_t r = fast_rand32(&seed);
        int j = (int)(r % (uint32_t)(i + 1));

        KV temp = S[i];
        S[i] = S[j];
        S[j] = temp;
    }
}

/* ---------------- Normal hash join ---------------- */

KV* build_global_hash_table(KV* R, int n, int* out_size)
{
    int table_size = next_pow2_int(n * 2);
    int mask = table_size - 1;

    KV* table = (KV*)malloc(sizeof(KV) * (size_t)table_size);

    if (table == NULL) {
        printf("Global hash table allocation failed for table_size=%d\n", table_size);
        exit(1);
    }

    for (int i = 0; i < table_size; i++) {
        table[i].key = EMPTY_KEY;
        table[i].value = 0;
    }

    for (int i = 0; i < n; i++) {
        int key = R[i].key;
        int h = local_hash(key, mask);

        while (table[h].key != EMPTY_KEY) {
            h = (h + 1) & mask;
        }

        table[h] = R[i];
    }

    *out_size = table_size;
    return table;
}

long long join_hash_baseline(KV* R, KV* S, int n)
{
    int table_size;
    KV* table = build_global_hash_table(R, n, &table_size);
    int mask = table_size - 1;

    long long matches = 0;

    for (int i = 0; i < n; i++) {
        int key = S[i].key;
        int h = local_hash(key, mask);

        while (1) {
            int k = table[h].key;

            if (k == EMPTY_KEY) {
                break;
            }

            if (k == key) {
                matches++;
                break;
            }

            h = (h + 1) & mask;
        }
    }

    free(table);
    return matches;
}

/* ---------------- Radix partitioning ---------------- */

void radix_partition(KV* input,
                     KV* output,
                     int n,
                     int* counts,
                     int* offsets)
{
    for (int p = 0; p < NUM_PARTITIONS; p++) {
        counts[p] = 0;
    }

    /*
       Count partition sizes.
    */
    for (int i = 0; i < n; i++) {
        int p = get_partition_id(input[i].key);
        counts[p]++;
    }

    /*
       Prefix sum.
    */
    offsets[0] = 0;

    for (int p = 1; p < NUM_PARTITIONS; p++) {
        offsets[p] = offsets[p - 1] + counts[p - 1];
    }

    /*
       Scatter into output.
    */
    int cursor[NUM_PARTITIONS];

    for (int p = 0; p < NUM_PARTITIONS; p++) {
        cursor[p] = offsets[p];
    }

    for (int i = 0; i < n; i++) {
        int p = get_partition_id(input[i].key);
        output[cursor[p]++] = input[i];
    }
}

/* ---------------- Fixed radix-partitioned SIMD hash join ---------------- */

long long join_radix_partitioned_hash_simd(KV* R, KV* S, int n)
{
    KV* R_part = (KV*)malloc(sizeof(KV) * (size_t)n);
    KV* S_part = (KV*)malloc(sizeof(KV) * (size_t)n);

    if (R_part == NULL || S_part == NULL) {
        printf("Partition buffer allocation failed for n=%d\n", n);
        free(R_part);
        free(S_part);
        exit(1);
    }

    int R_count[NUM_PARTITIONS];
    int S_count[NUM_PARTITIONS];

    int R_offset[NUM_PARTITIONS];
    int S_offset[NUM_PARTITIONS];

    radix_partition(R, R_part, n, R_count, R_offset);
    radix_partition(S, S_part, n, S_count, S_offset);

    /*
       Precompute maximum local hash table size so we allocate keys[] once.
       This avoids malloc/free inside every partition.
    */
    int max_r_count = 0;

    for (int p = 0; p < NUM_PARTITIONS; p++) {
        if (R_count[p] > max_r_count) {
            max_r_count = R_count[p];
        }
    }

    int max_table_size = next_pow2_int(max_r_count * 2 + 1);

    int* keys = (int*)malloc(sizeof(int) * (size_t)(max_table_size + SIMD_WIDTH));

    if (keys == NULL) {
        printf("Reusable SIMD key table allocation failed, max_table_size=%d\n",
               max_table_size);
        free(R_part);
        free(S_part);
        exit(1);
    }

    long long matches = 0;

    __m256i empty_vec = _mm256_set1_epi32(EMPTY_KEY);

    for (int p = 0; p < NUM_PARTITIONS; p++) {

        int r_count = R_count[p];
        int s_count = S_count[p];

        if (r_count == 0 || s_count == 0) {
            continue;
        }

        KV* Rp = &R_part[R_offset[p]];
        KV* Sp = &S_part[S_offset[p]];

        int table_size = next_pow2_int(r_count * 2 + 1);
        int mask = table_size - 1;

        /*
           Initialize only the portion used by this partition.
        */
        for (int i = 0; i < table_size + SIMD_WIDTH; i++) {
            keys[i] = EMPTY_KEY;
        }

        /*
           Build local flat key table.

           Important:
           Local table placement uses local_hash(), which is independent
           of the partition hash.
        */
        for (int i = 0; i < r_count; i++) {
            int key = Rp[i].key;
            int h = local_hash(key, mask);

            while (keys[h] != EMPTY_KEY) {
                h = (h + 1) & mask;
            }

            keys[h] = key;
        }

        /*
           Padding for safe SIMD loads near the end.

           If h is near table_size - 1, loading 8 ints from &keys[h]
           would read beyond table_size. The extra SIMD_WIDTH slots are
           valid memory and contain the wraparound beginning of the table.
        */
        for (int i = 0; i < SIMD_WIDTH; i++) {
            keys[table_size + i] = keys[i];
        }

        /*
           Probe local table with 8-wide SIMD block checks.
        */
        for (int i = 0; i < s_count; i++) {

            int key = Sp[i].key;
            int h = local_hash(key, mask);

            __m256i key_vec = _mm256_set1_epi32(key);

            while (1) {
                __m256i table_vec =
                    _mm256_loadu_si256((const __m256i*)&keys[h]);

                __m256i cmp =
                    _mm256_cmpeq_epi32(key_vec, table_vec);

                int match_mask =
                    _mm256_movemask_epi8(cmp);

                if (match_mask != 0) {
                    matches++;
                    break;
                }

                __m256i empty_cmp =
                    _mm256_cmpeq_epi32(table_vec, empty_vec);

                int empty_mask =
                    _mm256_movemask_epi8(empty_cmp);

                if (empty_mask != 0) {
                    break;
                }

                h = (h + SIMD_WIDTH) & mask;
            }
        }
    }

    free(keys);
    free(R_part);
    free(S_part);

    return matches;
}

/* ---------------- Memory reporting ---------------- */

double get_hash_memory_MB(int n)
{
    int table_size = next_pow2_int(n * 2);
    double bytes = (double)sizeof(KV) * table_size;
    return bytes / (1024.0 * 1024.0);
}

double get_radix_memory_MB(int n)
{
    /*
       Approximate peak internal memory:

       R_part + S_part + largest local keys[].

       This excludes input R and S because both algorithms receive those.
    */
    int approx_partition_size = (n + NUM_PARTITIONS - 1) / NUM_PARTITIONS;
    int local_table_size = next_pow2_int(approx_partition_size * 2 + 1);

    double partition_bytes =
        2.0 * (double)sizeof(KV) * (double)n;

    double local_key_bytes =
        (double)sizeof(int) * (double)(local_table_size + SIMD_WIDTH);

    return (partition_bytes + local_key_bytes) / (1024.0 * 1024.0);
}

/* ---------------- Experiment driver ---------------- */

void do_work(void)
{
    int sizes[] = {
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

    int tests = sizeof(sizes) / sizeof(sizes[0]);

    printf("Size,radix_time,hash_time,radix_matches,hash_matches,radix_memMB,hash_memMB\n");

    for (int t = 0; t < tests; t++) {

        int n = sizes[t];

        KV* R = (KV*)malloc(sizeof(KV) * (size_t)n);
        KV* S = (KV*)malloc(sizeof(KV) * (size_t)n);

        if (R == NULL || S == NULL) {
            printf("Input allocation failed for n=%d\n", n);
            free(R);
            free(S);
            return;
        }

        gen(R, S, n);

        flush_cache();

        double r1 = now();
        long long radix_matches = join_radix_partitioned_hash_simd(R, S, n);
        double r2 = now();

        flush_cache();

        double h1 = now();
        long long hash_matches = join_hash_baseline(R, S, n);
        double h2 = now();

        printf("%d,%.6f,%.6f,%lld,%lld,%.2f,%.2f\n",
               n,
               r2 - r1,
               h2 - h1,
               radix_matches,
               hash_matches,
               get_radix_memory_MB(n),
               get_hash_memory_MB(n));

        free(R);
        free(S);
    }
}

int main(void)
{
    init_flush_buffer();

    /*
       Run once or multiple times.
       If you want averaging, run multiple times and group by Size in Excel.
    */
    do_work();

    free(flush_buffer);
    return 0;
}
