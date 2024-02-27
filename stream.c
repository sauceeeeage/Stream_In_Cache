/*-----------------------------------------------------------------------*/
/* Program: STREAM                                                       */
/* Revision: $Id: stream.c,v 5.10 2013/01/17 16:01:06 mccalpin Exp mccalpin $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2013: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*           "tuned STREAM benchmark results"                            */
/*           "based on a variant of the STREAM benchmark code"           */
/*         Other comparable, clear, and reasonable labelling is          */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/
#include <float.h>
// #include <intrin.h>
#include <cpuid.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <x86intrin.h>

/*-----------------------------------------------------------------------
 * INSTRUCTIONS:
 *
 *	1) STREAM requires different amounts of memory to run on different
 *           systems, depending on both the system cache size(s) and the
 *           granularity of the system timer.
 *     You should adjust the value of 'STREAM_ARRAY_SIZE' (below)
 *           to meet *both* of the following criteria:
 *       (a) Each array must be at least 4 times the size of the
 *           available cache memory. I don't worry about the difference
 *           between 10^6 and 2^20, so in practice the minimum array size
 *           is about 3.8 times the cache size.
 *           Example 1: One Xeon E3 with 8 MB L3 cache
 *               STREAM_ARRAY_SIZE should be >= 4 million, giving
 *               an array size of 30.5 MB and a total memory requirement
 *               of 91.5 MB.
 *           Example 2: Two Xeon E5's with 20 MB L3 cache each (using OpenMP)
 *               STREAM_ARRAY_SIZE should be >= 20 million, giving
 *               an array size of 153 MB and a total memory requirement
 *               of 458 MB.
 *       (b) The size should be large enough so that the 'timing calibration'
 *           output by the program is at least 20 clock-ticks.
 *           Example: most versions of Windows have a 10 millisecond timer
 *               granularity.  20 "ticks" at 10 ms/tic is 200 milliseconds.
 *               If the chip is capable of 10 GB/s, it moves 2 GB in 200 msec.
 *               This means the each array must be at least 1 GB, or 128M
 *elements.
 *
 *      Version 5.10 increases the default array size from 2 million
 *          elements to 10 million elements in response to the increasing
 *          size of L3 caches.  The new default size is large enough for caches
 *          up to 20 MB.
 *      Version 5.10 changes the loop index variables from "register int"
 *          to "ssize_t", which allows array indices >2^32 (4 billion)
 *          on properly configured 64-bit systems.  Additional compiler options
 *          (such as "-mcmodel=medium") may be required for large memory runs.
 *
 *      Array size can be set at compile time without modifying the source
 *          code for the (many) compilers that support preprocessor definitions
 *          on the compile line.  E.g.,
 *                gcc -O -DSTREAM_ARRAY_SIZE=100000000 stream.c -o stream.100M
 *          will override the default size of 10M with a new size of 100M
 *elements per array.
 */
// #ifndef STREAM_ARRAY_SIZE
// #define STREAM_ARRAY_SIZE 16777216
// #endif
// 2^24

#ifndef HALF_CACHE_SIZE
#define HALF_CACHE_SIZE 67108864
#endif
// 1 GB

#ifndef STRIDE
#define STRIDE 1000
#endif

/*  2) STREAM runs each kernel "NTIMES" times and reports the *best* result
 *         for any iteration after the first, therefore the minimum value
 *         for NTIMES is 2.
 *      There are no rules on maximum allowable values for NTIMES, but
 *         values larger than the default are unlikely to noticeably
 *         increase the reported performance.
 *      NTIMES can also be set on the compile line without changing the source
 *         code using, for example, "-DNTIMES=7".
 */
#ifdef NTIMES
#if NTIMES <= 1
#define NTIMES 10
#endif
#endif
#ifndef NTIMES
#define NTIMES 10
#endif

/*  Users are allowed to modify the "OFFSET" variable, which *may* change the
 *         relative alignment of the arrays (though compilers may change the
 *         effective offset by making the arrays non-contiguous on some
 * systems). Use of non-zero values for OFFSET can be especially helpful if the
 *         STREAM_ARRAY_SIZE is set to a value close to a large power of 2.
 *      OFFSET can also be set on the compile line without changing the source
 *         code using, for example, "-DOFFSET=56".
 */
#ifndef OFFSET
#define OFFSET 0
#endif

/*
 *	3) Compile the code with optimization.  Many compilers generate
 *       unreasonably bad code before the optimizer tightens things up.
 *     If the results are unreasonably good, on the other hand, the
 *       optimizer might be too smart for me!
 *
 *     For a simple single-core version, try compiling with:
 *            cc -O stream.c -o streamcounter
 *     This is known to work on many, many systems....
 *
 *     To use multiple cores, you need to tell the compiler to obey the OpenMP
 *       directives in the code.  This varies by compiler, but a common example
 *is gcc -O -fopenmp stream.c -o stream_omp The environment variable
 *OMP_NUM_THREADS allows runtime control of the number of threads/cores used
 *when the resulting "stream_omp" program is executed.
 *
 *     To run with single-precision variables and arithmetic, simply add
 *         -DSTREAM_TYPE=float
 *     to the compile line.
 *     Note that this changes the minimum array sizes required --- see (1)
 *above.
 *
 *     The preprocessor directive "TUNED" does not do much -- it simply causes
 *the code to call separate functions to execute each kernel.  Trivial versions
 *       of these functions are provided, but they are *not* tuned -- they just
 *       provide predefined interfaces to be replaced with tuned code.
 *
 *
 *	4) Optional: Mail the results to mccalpin@cs.virginia.edu
 *	   Be sure to include info that will help me understand:
 *		a) the computer hardware configuration (e.g., processor model,
 *memory type) b) the compiler name/version and compilation flags c) any
 *run-time information (such as OMP_NUM_THREADS) d) all of the output from the
 *test case.
 *
 * Thanks!
 *
 *-----------------------------------------------------------------------*/

#define HLINE "-------------------------------------------------------------\n"

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

// static STREAM_TYPE a[STREAM_ARRAY_SIZE + OFFSET],
//     b[STREAM_ARRAY_SIZE + OFFSET],
//     c[STREAM_ARRAY_SIZE + OFFSET];

static double t1[HALF_CACHE_SIZE], t2[HALF_CACHE_SIZE];

static double avgtime[5] = {0}, maxtime[5] = {0},
              mintime[5] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};

static char *label[5] = {
    "CYCLIC:                   ", "SAWTOOTH:                 ",
    "RAND FORWARD FORWARD:     ", "RAND FORWARD BACKWARD:    ",
    "RAND BACKWARD BACKWARD:   "};

extern double mysecond();

extern void checkSTREAMresults();

#ifdef TUNED
extern void tuned_STREAM_Copy();
extern void tuned_STREAM_Scale(STREAM_TYPE scalar);
extern void tuned_STREAM_Add();
extern void tuned_STREAM_Triad(STREAM_TYPE scalar);
#endif
#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

long to_next_power_of_two(long x) {
  x--;
  x |= x >> 1;  // handle  2 bit numbers
  x |= x >> 2;  // handle  4 bit numbers
  x |= x >> 4;  // handle  8 bit numbers
  x |= x >> 8;  // handle 16 bit numbers
  x |= x >> 16; // handle 32 bit numbers
  x++;
  return x;
}

/*
 * more precisely, the function f(k) = T_k mod 2^m is a permutation on { 0, 1,
 * ..., 2^m-1 }, where T_k is the k-th triangular number which is defined as T_k
 * = k*(k+1)/2.
 * https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
 */
long triangular_number_mod_2n(long k, long power_of_two) {
  long triangular_number =
      k * (k + 1) / 2; // don't know if we need to bitwise this to speed it up
                       // or will it mess up the compiler optimization?
  return triangular_number % power_of_two;
}

static inline void clear_cache() {
  double tmp;
  for (int t = 0; t < HALF_CACHE_SIZE; t++) {
    tmp = t1[t];
    tmp = t2[t];
  }
}

/* use rdtscp and cpuid to get a high resolution time stamp, and if rdtscp is
 * not available, use rdtsc
 */

static inline unsigned long long rdtsc() {
  // unsigned int lo, hi;
  // __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  // return ((unsigned long long)hi << 32) | lo;
  unsigned long long i;
  unsigned int ui;
  i = __rdtscp(&ui);
  return i;
}

int rdtscp_supported(void) {
  unsigned a, b, c, d;
  if (__get_cpuid(0x80000001, &a, &b, &c, &d) && (d & (1 << 27))) {
    // RDTSCP is supported.
    return 1;
  } else {
    // RDTSCP is not supported.
    return 0;
  }
}

int main(int argc, char *argv[]) {
  unsigned long long cycles[5][NTIMES];
  int quantum, checktick();
  int BytesPerWord;
  int k;
  ssize_t j;
  STREAM_TYPE scalar;
  double t, times[5][NTIMES];
  size_t array_size = atoi(argv[1]);
  cpu_set_t mask;

  double bytes[5] = {3 * sizeof(STREAM_TYPE) * array_size,
                     3 * sizeof(STREAM_TYPE) * array_size,
                     3 * sizeof(STREAM_TYPE) * array_size,
                     3 * sizeof(STREAM_TYPE) * array_size,
                     3 * sizeof(STREAM_TYPE) * array_size};

  STREAM_TYPE *cyclic_a, *cyclic_b, *cyclic_c, *cyclic;
  STREAM_TYPE *sawtooth_a, *sawtooth_b, *sawtooth_c, *sawtooth;
  STREAM_TYPE *rand_forward_forward_a, *rand_forward_forward_b,
      *rand_forward_forward_c, *rand_forward_forward;
  STREAM_TYPE *rand_forward_backward_a, *rand_forward_backward_b,
      *rand_forward_backward_c, *rand_forward_backward;
  STREAM_TYPE *rand_backward_backward_a, *rand_backward_backward_b,
      *rand_backward_backward_c, *rand_backward_backward;
  STREAM_TYPE *time_test;

  // __cpuid(cpuInfo, 0);

  /* SET THREAD AFFINITY */
#define _GNU_SOURCE 1
#define __USE_GNU 1

  CPU_ZERO(&mask);
  CPU_SET(0, &mask);
  if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
    printf("Failed to set CPU affinity\n");
    exit(1);
  } else {
    int cpu = sched_getcpu();
    printf("CPU affinity set to CPU %d\n", cpu);
  }

  /* --- SETUP --- determine precision and check timing --- */

  printf(HLINE);
  printf("STREAM version $Revision: 5.10 $\n");
  printf(HLINE);
  BytesPerWord = sizeof(STREAM_TYPE);
  printf("This system uses %d bytes per array element.\n", BytesPerWord);

  printf(HLINE);
  if (rdtscp_supported()) {
    printf("rdtscp is supported\n");
  } else {
    printf("rdtscp is not supported\n");
    exit(1);
  }

#ifdef N
  printf("*****  WARNING: ******\n");
  printf("      It appears that you set the preprocessor variable N when "
         "compiling this code.\n");
  printf("      This version of the code uses the preprocessor variable "
         "STREAM_ARRAY_SIZE to control the array size\n");
  printf("      Reverting to default value of STREAM_ARRAY_SIZE=%llu\n",
         (unsigned long long)STREAM_ARRAY_SIZE);
  printf("*****  WARNING: ******\n");
#endif

  printf("Array size = %llu (elements), Offset = %d (elements)\n",
         (unsigned long long)array_size, OFFSET);
  printf("Memory per array = %.1f MiB (= %.1f GiB).\n",
         BytesPerWord * ((double)array_size / 1024.0 / 1024.0),
         BytesPerWord * ((double)array_size / 1024.0 / 1024.0 / 1024.0));
  printf("Total memory required = %.1f MiB (= %.1f GiB).\n",
         (3.0 * BytesPerWord) * ((double)array_size / 1024.0 / 1024.),
         (3.0 * BytesPerWord) * ((double)array_size / 1024.0 / 1024. / 1024.));
  printf("Each kernel will be executed %d times.\n", NTIMES);
  printf(" The *best* time for each kernel (excluding the first iteration)\n");
  printf(" will be used to compute the reported bandwidth.\n");

#ifdef _OPENMP
  printf(HLINE);
#pragma omp parallel
  {
#pragma omp master
    {
      k = omp_get_num_threads();
      printf("Number of Threads requested = %i\n", k);
    }
  }
#endif

#ifdef _OPENMP
  k = 0;
#pragma omp parallel
#pragma omp atomic
  k++;
  printf("Number of Threads counted = %i\n", k);
#endif

  /* Get initial value for system clock. */
  for (int t = 0; t < HALF_CACHE_SIZE; t++) {
    t1[t] = 1.0 * t;
    t2[t] = 2.0 * t;
  }

  printf(HLINE);

  if ((quantum = checktick()) >= 1)
    printf("Your clock granularity/precision appears to be "
           "%d nanoseconds.\n",
           quantum);
  else {
    printf("Your clock granularity appears to be "
           "less than one nanosecond.\n");
    quantum = 1;
  }

  t = rdtsc();
  time_test = (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  if (time_test == NULL) {
    printf("Failed to allocate memory for time_test array\n");
    exit(1);
  }
#pragma omp parallel for
  for (int i = 0; i < array_size; i++) {
    time_test[i] = 1.0;
  }
#pragma omp parallel for
  for (j = 0; j < array_size; j++)
    time_test[j] = 2.0E0 * time_test[j];
  t = (rdtsc() - t);

  printf("Each test below will take on the order"
         " of %d cycles.\n",
         (int)t);
  printf("   (= %d clock ticks)\n", (int)(t / quantum));
  printf("Increase the size of the arrays if this shows that\n");
  printf("you are not getting at least 20 clock ticks per test.\n");

  printf(HLINE);

  printf("WARNING -- The above is only a rough guideline.\n");
  printf("For best results, please be sure you know the\n");
  printf("precision of your system timer.\n");
  printf(HLINE);

  /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

  scalar = 3.0;

  /*----------------------CYCLIC---------------------------*/

  // cyclic_a = (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  // cyclic_b = (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  // cyclic_c = (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  cyclic = (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);

  // if (cyclic_a == NULL || cyclic_b == NULL || cyclic_c == NULL) {
  //   printf("Failed to allocate memory for cyclic arrays\n");
  //   exit(1);
  // }

  if (cyclic == NULL) {
    printf("Failed to allocate memory for cyclic arrays\n");
    exit(1);
  }

#pragma omp parallel for
  for (j = 0; j < array_size; j++) {
    // cyclic_a[j] = 1.0;
    // cyclic_b[j] = 2.0;
    // cyclic_c[j] = 0.0;
    cyclic[j] = 1.0 * j;
  }
  STREAM_TYPE cyclic_tmp;
  clear_cache();

  // load the array into the cache

  for (j = 0; j < array_size; j += 1) {
    cyclic_tmp = cyclic[j];
  }

#pragma omp parallel for
  for (int k = 0; k < NTIMES; k++) {
    // times[0][k] = mysecond(); // start
    cycles[0][k] = rdtsc();
#ifdef TUNED
    tuned_STREAM_Copy();
#else
    // CYCLIC
    for (j = 0; j < array_size; j += 8) {
      cyclic_tmp = cyclic[j];
    }
    for (j = 0; j < array_size; j += 8) {
      cyclic_tmp = cyclic[j];
    }
#endif
    // times[0][k] = (mysecond() - times[0][k]) / 2; // end
    cycles[0][k] = (rdtsc() - cycles[0][k]) / 2;
  }

  free(cyclic);
  // checkSTREAMresults(cyclic_a, cyclic_b, cyclic_c, "CYCLIC", array_size);
  // free(cyclic_a);
  // free(cyclic_b);
  // free(cyclic_c);
  /*----------------------SAWTOOTH---------------------------*/

  // sawtooth_a = (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  // sawtooth_b = (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  // sawtooth_c = (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  sawtooth = (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);

  // if (sawtooth_a == NULL || sawtooth_b == NULL || sawtooth_c == NULL) {
  //   printf("Failed to allocate memory for cyclic arrays\n");
  //   exit(1);
  // }
  if (sawtooth == NULL) {
    printf("Failed to allocate memory for cyclic arrays\n");
    exit(1);
  }
#pragma omp parallel for
  for (j = 0; j < array_size; j++) {
    // sawtooth_a[j] = 1.0;
    // sawtooth_b[j] = 2.0;
    // sawtooth_c[j] = 0.0;
    sawtooth[j] = 1.0 * j;
  }
  STREAM_TYPE sawtooth_tmp;
  clear_cache();
  for (j = 0; j < array_size; j += 1) {
    sawtooth_tmp = sawtooth[j];
  }
#pragma omp parallel for
  for (int k = 0; k < NTIMES; k++) {
    // times[1][k] = mysecond(); // start
    cycles[1][k] = rdtsc();
#ifdef TUNED
    tuned_STREAM_Scale(scalar);
#else
    // SAWTOOTH
    for (j = 0; j < array_size; j += 8) {
      sawtooth_tmp = sawtooth[j];
    }
    for (j = array_size - 1; j >= 0; j -= 8) {
      sawtooth_tmp = sawtooth[j];
    }
#endif
    // times[1][k] = (mysecond() - times[1][k]) / 2; // end
    cycles[1][k] = (rdtsc() - cycles[1][k]) / 2;
  }
  free(sawtooth);
  // checkSTREAMresults(sawtooth_a, sawtooth_b, sawtooth_c, "SAWTOOTH",
  //                    array_size);
  // free(sawtooth_a);
  // free(sawtooth_b);
  // free(sawtooth_c);
  /*----------------------RAND FORWARD FORWARD---------------------------*/

  // rand_forward_forward_a =
  //     (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  // rand_forward_forward_b =
  //     (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  // rand_forward_forward_c =
  //     (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);

  // if (rand_forward_forward_a == NULL || rand_forward_forward_b == NULL ||
  //     rand_forward_forward_c == NULL) {
  //   printf("Failed to allocate memory for cyclic arrays\n");
  //   exit(1);
  // }
  rand_forward_forward =
      (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);

  if (rand_forward_forward == NULL) {
    printf("Failed to allocate memory for cyclic arrays\n");
    exit(1);
  }

#pragma omp parallel for
  for (j = 0; j < array_size; j++) {
    // rand_forward_forward_a[j] = 1.0;
    // rand_forward_forward_b[j] = 2.0;
    // rand_forward_forward_c[j] = 0.0;
    rand_forward_forward[j] = 1.0 * j;
  }
  STREAM_TYPE rand_forward_forward_tmp;
  clear_cache();
  for (j = 0; j < array_size; j += 1) {
    rand_forward_forward_tmp = rand_forward_forward[j];
  }
#pragma omp parallel for
  for (int k = 0; k < NTIMES; k++) {
    // times[2][k] = mysecond(); // start
    cycles[2][k] = rdtsc();
#ifdef TUNED
    tuned_STREAM_Add();
#else
    // CYCLIC(forward-forward) + pseudo random accesses for the loop access
    // pattern don't think we need an actual random number to start with, but it
    // may be better??
    for (int p = 0, stride = 0, rand = 0; p < array_size; p += 8) {
      rand = (rand + stride) & (array_size - 1);
      rand_forward_forward_tmp = rand_forward_forward[rand];
      stride += 8;
    }
    for (int p = 0, stride = 0, rand = 0; p < array_size; p += 8) {
      rand = (rand + stride) & (array_size - 1);
      rand_forward_forward_tmp = rand_forward_forward[rand];
      stride += 8;
    }
#endif
    // times[2][k] = (mysecond() - times[2][k]) / 2; // end
    cycles[2][k] = (rdtsc() - cycles[2][k]) / 2;
  }
  free(rand_forward_forward);
  // checkSTREAMresults(rand_forward_forward_a, rand_forward_forward_b,
  //                    rand_forward_forward_c, "RAND FORWARD FORWARD",
  //                    array_size);
  // free(rand_forward_forward_a);
  // free(rand_forward_forward_b);
  // free(rand_forward_forward_c);
  /*----------------------RAND FORWARD BACKWARD---------------------------*/

  // rand_forward_backward_a =
  //     (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  // rand_forward_backward_b =
  //     (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  // rand_forward_backward_c =
  //     (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);

  // if (rand_forward_backward_a == NULL || rand_forward_backward_b == NULL ||
  //     rand_forward_backward_c == NULL) {
  //   printf("Failed to allocate memory for cyclic arrays\n");
  //   exit(1);
  // }
  rand_forward_backward =
      (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  if (rand_forward_backward == NULL) {
    printf("Failed to allocate memory for cyclic arrays\n");
    exit(1);
  }

#pragma omp parallel for
  for (j = 0; j < array_size; j++) {
    // rand_forward_backward_a[j] = 1.0;
    // rand_forward_backward_b[j] = 2.0;
    // rand_forward_backward_c[j] = 0.0;
    rand_forward_backward[j] = 1.0 * j;
  }
  STREAM_TYPE rand_forward_backward_tmp;
  clear_cache();
  for (j = 0; j < array_size; j += 1) {
    rand_forward_backward_tmp = rand_forward_backward[j];
  }
#pragma omp parallel for
  for (int k = 0; k < NTIMES; k++) {
    int for_back_stride = 0, for_back_rand = 0;
    // times[3][k] = mysecond(); // start timing
    cycles[3][k] = rdtsc();
#ifdef TUNED
    tuned_STREAM_Triad(scalar);
#else
    // SAWTOOTH(forward-backward) + pseudo random accesses for the loop access
    // pattern REAL SAWTOOTH
    for (int p = 0; p < array_size; p += 8) {
      for_back_rand = (for_back_rand + for_back_stride) & (array_size - 1);
      // rand_forward_backward_a[for_back_rand] =
      //     rand_forward_backward_b[for_back_rand] +
      //     scalar * rand_forward_backward_c[for_back_rand];
      rand_forward_backward_tmp = rand_forward_backward[for_back_rand];
      for_back_stride += 8;
    }
    for (int p = 0; p < array_size; p += 8) {
      for_back_rand = (for_back_rand - for_back_stride) & (array_size - 1);
      // rand_forward_backward_a[for_back_rand] =
      //     rand_forward_backward_b[for_back_rand] +
      //     scalar * rand_forward_backward_c[for_back_rand];
      rand_forward_backward_tmp = rand_forward_backward[for_back_rand];
      for_back_stride -= 8;
    }

    // END OF THE REAL SAWTOOTH

    // GNU C
    // asm("":::"memory"); // this is compiler fence
    // asm volatile ("mfence" ::: "memory"); // this is hardware fence

    // FAKE SAWTOOTH
    // for (j=0; j<=STREAM_ARRAY_SIZE-STRIDE; j+=STRIDE){
    // 	for (int u = 0; u < STRIDE/2; u++) {
    // 		a[j+u] = b[j+u]+scalar*c[j+u];
    // 	}
    // 	for (int u = STRIDE-1; u >= STRIDE/2; u--) {
    // 		a[j+u] = b[j+u]+scalar*c[j+u];
    // 	}
    // }
    // END OF THE FAKE SAWTOOTH
#endif
    // times[3][k] = (mysecond() - times[3][k]) / 2; // end
    cycles[3][k] = (rdtsc() - cycles[3][k]) / 2;
  }
  free(rand_forward_backward);
  // checkSTREAMresults(rand_forward_backward_a, rand_forward_backward_b,
  //                    rand_forward_backward_c, "RAND FORWARD BACKWARD",
  //                    array_size);
  // free(rand_forward_backward_a);
  // free(rand_forward_backward_b);
  // free(rand_forward_backward_c);
  /*----------------------RAND BACKWARD BACKWARD---------------------------*/

  // rand_backward_backward_a =
  //     (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  // rand_backward_backward_b =
  //     (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  // rand_backward_backward_c =
  //     (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);

  // if (rand_backward_backward_a == NULL || rand_backward_backward_b == NULL ||
  //     rand_backward_backward_c == NULL) {
  //   printf("Failed to allocate memory for cyclic arrays\n");
  //   exit(1);
  // }
  rand_backward_backward =
      (STREAM_TYPE *)malloc(sizeof(STREAM_TYPE) * array_size);
  if (rand_backward_backward == NULL) {
    printf("Failed to allocate memory for cyclic arrays\n");
    exit(1);
  }

#pragma omp parallel for
  for (j = 0; j < array_size; j++) {
    // rand_backward_backward_a[j] = 1.0;
    // rand_backward_backward_b[j] = 2.0;
    // rand_backward_backward_c[j] = 0.0;
    rand_backward_backward[j] = 1.0 * j;
  }
  STREAM_TYPE rand_backward_backward_tmp;
  clear_cache();
  for (j = 0; j < array_size; j += 1) {
    rand_backward_backward_tmp = rand_backward_backward[j];
  }
#pragma omp parallel for
  for (int k = 0; k < NTIMES; k++) {
    // times[4][k] = mysecond(); // start
    cycles[4][k] = rdtsc();
#ifdef TUNED
    tuned_STREAM_Add();
#else
    // CYCLIC(backward-backward) + pseudo random accesses for the loop access
    // pattern
    for (int p = 0, stride = array_size, rand = array_size / 2; p < array_size;
         p += 8) {
      rand = (rand - stride) & (array_size - 1);
      rand_backward_backward_tmp = rand_backward_backward[rand];
      stride -= 8;
    }
    for (int p = 0, stride = array_size, rand = array_size / 2; p < array_size;
         p += 8) {
      rand = (rand - stride) & (array_size - 1);
      rand_backward_backward_tmp = rand_backward_backward[rand];
      stride -= 8;
    }
#endif
    // times[4][k] = (mysecond() - times[4][k]) / 2; // end
    cycles[4][k] = (rdtsc() - cycles[4][k]) / 2;
  }
  free(rand_backward_backward);
  // checkSTREAMresults(rand_backward_backward_a, rand_backward_backward_b,
  //                    rand_backward_backward_c, "RAND BACKWARD BACKWARD",
  //                    array_size);
  // free(rand_backward_backward_a);
  // free(rand_backward_backward_b);
  // free(rand_backward_backward_c);

  /*	--- SUMMARY --- */
  // convert cycles to nanoseconds
  double frequency = 5.76 * 1000 * 1000 * 1000;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < NTIMES; j++) {
      // times[i][j] = cycles[i][j] / frequency;
      times[i][j] = (double)cycles[i][j];
    }
  }

  for (k = 0; k < NTIMES; k++) /* note -- skip first iteration */
  {
    for (j = 0; j < 5; j++) {
      avgtime[j] = avgtime[j] + times[j][k];
      mintime[j] = MIN(mintime[j], times[j][k]);
      maxtime[j] = MAX(maxtime[j], times[j][k]);
    }
  }

  printf("Function                  Best Rate MiB/cycles  Avg cycles      Min "
         "cycles     "
         " Max cycles      Access Times      Avg Cycles per Access\n");
  for (j = 0; j < 5; j++) {
    avgtime[j] = avgtime[j] / (double)(NTIMES - 1);

    printf("%s%e  %e  %e  %e    %e        %e\n", label[j],
           1024 * 1024 * bytes[j] / mintime[j], avgtime[j], mintime[j],
           maxtime[j], (double)(array_size / 8),
           (avgtime[j] / (double)(array_size / 8)));
  }

  return 0;
}

#define M 20

int checktick() {
  int i, minDelta, Delta;
  double t1, t2, timesfound[M];

  /*  Collect a sequence of M unique time values from the system. */

  for (i = 0; i < M; i++) {
    t1 = mysecond();
    while (((t2 = mysecond()) - t1) < 1.0E-6)
      ;
    timesfound[i] = t1 = t2;
  }

  /*
   * Determine the minimum difference between these M values.
   * This result will be our estimate (in nanoseconds) for the
   * clock granularity.
   */

  minDelta = 1000000;
  for (i = 1; i < M; i++) {
    Delta = (int)(1.0E6 * (timesfound[i] - timesfound[i - 1]));
    minDelta = MIN(minDelta, MAX(Delta, 0));
  }

  return (minDelta);
}

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <time.h>

inline double mysecond() {
  struct timespec ts;
  int i;

  i = clock_gettime(CLOCK_MONOTONIC, &ts);
  return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif

void checkSTREAMresults(STREAM_TYPE *a, STREAM_TYPE *b, STREAM_TYPE *c,
                        char *label, size_t array_size) {
  STREAM_TYPE aj, bj, cj, t1j, t2j, scalar;
  STREAM_TYPE aSumErr, bSumErr, cSumErr, t1jSumErr, t2jSumErr;
  STREAM_TYPE aAvgErr, bAvgErr, cAvgErr, t1jAvgErr, t2jAvgErr;
  double epsilon;
  ssize_t j;
  int k, ierr, err;

  /* reproduce initialization */
  aj = 1.0;
  bj = 2.0;
  cj = 0.0;
  t1j = 1.0;
  t2j = 2.0;
  /* a[] is modified during timing check */
  aj = 2.0E0 * aj;
  /* now execute timed loop */
  scalar = 3.0;
  for (k = 0; k < NTIMES; k += 8) {
    //        cj = aj;
    //        bj = scalar * cj;
    //        cj = aj + bj;
    t1j = t2j * scalar;
    aj = bj + scalar * cj;
  }

  /* accumulate deltas between observed and expected results */
  aSumErr = 0.0;
  bSumErr = 0.0;
  cSumErr = 0.0;
  t1jSumErr = 0.0;
  t2jSumErr = 0.0;
  for (j = 0; j < array_size; j += 8) {
    aSumErr += abs(a[j] - aj);
    bSumErr += abs(b[j] - bj);
    cSumErr += abs(c[j] - cj);
    // if (j == 417) printf("Index 417: c[j]: %f, cj: %f\n",c[j],cj);	//
    // MCCALPIN
  }
  for (int t = 0; t < HALF_CACHE_SIZE; t++) {
    t1jSumErr += abs(t1[t] - t1j);
    t2jSumErr += abs(t2[t] - t2j);
  }
  aAvgErr = aSumErr / ((STREAM_TYPE)array_size / 8);
  bAvgErr = bSumErr / ((STREAM_TYPE)array_size / 8);
  cAvgErr = cSumErr / ((STREAM_TYPE)array_size / 8);

  if (sizeof(STREAM_TYPE) == 4) {
    epsilon = 1.e-6;
  } else if (sizeof(STREAM_TYPE) == 8) {
    epsilon = 1.e-13;
  } else {
    printf("WEIRD: sizeof(STREAM_TYPE) = %lu\n", sizeof(STREAM_TYPE));
    epsilon = 1.e-6;
  }

  err = 0;
  if (abs(aAvgErr / aj) > epsilon) {
    err++;
    printf("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",
           epsilon);
    printf("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", aj,
           aAvgErr, abs(aAvgErr) / aj);
    ierr = 0;
    for (j = 0; j < array_size; j++) {
      if (abs(a[j] / aj - 1.0) > epsilon) {
        ierr++;
#ifdef VERBOSE
        if (ierr < 10) {
          printf("         array a: index: %ld, expected: %e, observed: %e, "
                 "relative error: %e\n",
                 j, aj, a[j], abs((aj - a[j]) / aAvgErr));
        }
#endif
      }
    }
    printf("     For array a[], %d errors were found.\n", ierr);
  }
  if (abs(bAvgErr / bj) > epsilon) {
    err++;
    printf("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",
           epsilon);
    printf("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", bj,
           bAvgErr, abs(bAvgErr) / bj);
    printf("     AvgRelAbsErr > Epsilon (%e)\n", epsilon);
    ierr = 0;
    for (j = 0; j < array_size; j++) {
      if (abs(b[j] / bj - 1.0) > epsilon) {
        ierr++;
#ifdef VERBOSE
        if (ierr < 10) {
          printf("         array b: index: %ld, expected: %e, observed: %e, "
                 "relative error: %e\n",
                 j, bj, b[j], abs((bj - b[j]) / bAvgErr));
        }
#endif
      }
    }
    printf("     For array b[], %d errors were found.\n", ierr);
  }
  if (abs(cAvgErr / cj) > epsilon) {
    err++;
    printf("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",
           epsilon);
    printf("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n", cj,
           cAvgErr, abs(cAvgErr) / cj);
    printf("     AvgRelAbsErr > Epsilon (%e)\n", epsilon);
    ierr = 0;
    for (j = 0; j < array_size; j++) {
      if (abs(c[j] / cj - 1.0) > epsilon) {
        ierr++;
#ifdef VERBOSE
        if (ierr < 10) {
          printf("         array c: index: %ld, expected: %e, observed: %e, "
                 "relative error: %e\n",
                 j, cj, c[j], abs((cj - c[j]) / cAvgErr));
        }
#endif
      }
    }
    printf("     For array c[], %d errors were found.\n", ierr);
  }
  if (err == 0) {
    printf("Solution Validates for %s: avg error less than %e on all three "
           "arrays\n",
           label, epsilon);
  }

#ifdef VERBOSE
  printf("Results Validation Verbose Results: \n");
  printf("    Expected a(1), b(1), c(1): %f %f %f \n", aj, bj, cj);
  printf("    Observed a(1), b(1), c(1): %f %f %f \n", a[1], b[1], c[1]);
  printf("    Rel Errors on a, b, c:     %e %e %e \n", abs(aAvgErr / aj),
         abs(bAvgErr / bj), abs(cAvgErr / cj));
#endif
}

#ifdef TUNED
/* stubs for "tuned" versions of the kernels */
void tuned_STREAM_Copy() {
  ssize_t j;
#pragma omp parallel for
  for (j = 0; j < STREAM_ARRAY_SIZE; j++)
    c[j] = a[j];
}

void tuned_STREAM_Scale(STREAM_TYPE scalar) {
  ssize_t j;
#pragma omp parallel for
  for (j = 0; j < STREAM_ARRAY_SIZE; j++)
    b[j] = scalar * c[j];
}

void tuned_STREAM_Add() {
  ssize_t j;
#pragma omp parallel for
  for (j = 0; j < STREAM_ARRAY_SIZE; j++)
    c[j] = a[j] + b[j];
}

void tuned_STREAM_Triad(STREAM_TYPE scalar) {
  ssize_t j;
#pragma omp parallel for
  for (j = 0; j < STREAM_ARRAY_SIZE; j++)
    a[j] = b[j] + scalar * c[j];
}
/* end of stubs for the "tuned" versions of the kernels */
#endif
