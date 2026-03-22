#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// exercice 1
double dist(float *u, float *v, int n) {
  double sum = 0.;
  for (int i = 0; i < n; i++) {
    double ui = (double)u[i];
    double vi = (double)v[i];
    double squareroot =
        sqrt((ui * ui + (vi * vi)) / (1 + (ui * vi) * (ui * vi)));
    sum += squareroot;
  }
  return sum;
}

// exercice 2
double dist_avx(float *u, float *v, int n) {
  __m256d vec_sum = _mm256_setzero_pd();
  __m256d one = _mm256_set1_pd(1.0);

  for (int i = 0; i < n - 3; i += 4) {
    // load 4 floats and convert to double
    __m256d vec_u = _mm256_cvtps_pd(_mm_load_ps(&u[i]));
    __m256d vec_v = _mm256_cvtps_pd(_mm_load_ps(&v[i]));

    // sqrt
    __m256d vec_sqrt = _mm256_sqrt_pd(_mm256_div_pd(
        _mm256_add_pd(_mm256_mul_pd(vec_u, vec_u), _mm256_mul_pd(vec_v, vec_v)),
        _mm256_add_pd(one, _mm256_mul_pd(_mm256_mul_pd(vec_u, vec_v),
                                         _mm256_mul_pd(vec_u, vec_v)))));
    // accumulate sum
    vec_sum = _mm256_add_pd(vec_sum, vec_sqrt);
  }
  // horizontal add to sum the elements of vec_sum
  vec_sum = _mm256_hadd_pd(vec_sum, vec_sum);
  vec_sum = _mm256_add_pd(vec_sum, _mm256_permute4x64_pd(vec_sum, 0b00011011));

  // cast
  double result = ((double *)&vec_sum)[0];
  return result;
}

// exercice 3
double dist_avx_gen(float *u, float *v, int n) {
  __m256d vec_sum = _mm256_setzero_pd();
  __m256d one = _mm256_set1_pd(1.0);

  for (int i = 0; i < n - 3; i += 4) {
    // load 4 floats and convert to double
    __m256d vec_u = _mm256_cvtps_pd(_mm_loadu_ps(&u[i]));
    __m256d vec_v = _mm256_cvtps_pd(_mm_loadu_ps(&v[i]));

    // sqrt
    __m256d vec_sqrt = _mm256_sqrt_pd(_mm256_div_pd(
        _mm256_add_pd(_mm256_mul_pd(vec_u, vec_u), _mm256_mul_pd(vec_v, vec_v)),
        _mm256_add_pd(one, _mm256_mul_pd(_mm256_mul_pd(vec_u, vec_v),
                                         _mm256_mul_pd(vec_u, vec_v)))));
    // accumulate sum
    vec_sum = _mm256_add_pd(vec_sum, vec_sqrt);
  }
  // horizontal add to sum the elements of vec_sum
  vec_sum = _mm256_hadd_pd(vec_sum, vec_sum);
  vec_sum = _mm256_add_pd(vec_sum, _mm256_permute4x64_pd(vec_sum, 0b00011011));

  double result = ((double *)&vec_sum)[0];

  // handle remaining elements
  if (n % 4 != 0) {
    for (int i = n - (n % 4); i < n; i++) {
      double ui = (double)u[i];
      double vi = (double)v[i];
      double squareroot =
          sqrt((ui * ui + (vi * vi)) / (1 + (ui * vi) * (ui * vi)));
      result += squareroot;
    }
  }
  return result;
}

// exercice 4
double flex_dist_gen(float *u, float *v, int n, int a, int b, int mode) {
  int end = fmin(b, n);
  if (mode == 0) {
    return dist(&u[a], &v[a], end - a);
  } else if (mode == 1) {
    // avx mode
    return dist_avx_gen(&u[a], &v[a], end - a);
  } else {
    fprintf(stderr, "Invalid mode: %d\n", mode);
    return -1.0;
  }
}

// exercice 5
// structure to pass arguments to threads
struct Arguments {
  float *u;
  float *v;
  int n;
  int a;
  int b;
  int mode;
  int thread_id;
  double *result;
};

void *flex_dist_gen_vec(void *struct_ptr) {
  // function to be executed by each thread
  struct Arguments args = *(struct Arguments *)struct_ptr;
  float *u = args.u;
  float *v = args.v;
  int n = args.n;
  int a = args.a;
  int b = args.b;
  int mode = args.mode;
  int thread_id = args.thread_id;
  double *result = args.result;
  int end = fmin(b, n);
  result[thread_id] = flex_dist_gen(u, v, n, a, b, mode);
  pthread_exit(NULL);
}

double distPar(float *u, float *v, int n, int nb_threads, int mode) {
  pthread_t threads[nb_threads]; // array to hold thread IDs
  double *results;
  results = (double *)malloc(nb_threads * sizeof(double));
  struct Arguments *thread_args =
      (struct Arguments *)malloc(nb_threads * sizeof(struct Arguments));

  for (int i = 0; i < nb_threads; i++) {
    thread_args[i].u = u;
    thread_args[i].v = v;
    thread_args[i].n = n;
    thread_args[i].a = i * n / nb_threads;
    thread_args[i].b = (i + 1) * n / nb_threads;
    thread_args[i].mode = mode;
    thread_args[i].thread_id = i;
    thread_args[i].result = results;
    // create thread
    pthread_create(&threads[i], NULL, flex_dist_gen_vec,
                   (void *)&thread_args[i]);
  }

  /* Wait for all threads to complete */
  double sum = 0;
  for (int i = 0; i < nb_threads; i++) {
    pthread_join(threads[i], NULL);
    sum += results[i];
  }
  free(results);
  free(thread_args);
  return sum;
}

int main() {
#define N 1024 * 1024
  // heap allocation
  float *u = (float *)aligned_alloc(32, N * sizeof(float));
  float *v = (float *)aligned_alloc(32, N * sizeof(float));

  for (int i = 0; i < N; i++) {
    // random float between 0 and 1
    u[i] = (float)rand() / (float)(RAND_MAX);
    v[i] = (float)rand() / (float)(RAND_MAX);
  }

  printf("Calculating distances...\n");
  struct timeval t1, t2, t3, t4, t5;
  gettimeofday(&t1, 0);
  double dist_scalar = dist(u, v, N);
  gettimeofday(&t2, 0);
  double dist_vectoriel = dist_avx_gen(u, v, N);
  gettimeofday(&t3, 0);
  double dist_multithread = distPar(u, v, N, 8, 0);
  gettimeofday(&t4, 0);
  double dist_multithread_vectoriel = distPar(u, v, N, 8, 1);
  gettimeofday(&t5, 0);

  double duration_scalar = (double)(t2.tv_sec - t1.tv_sec) +
                           (double)(t2.tv_usec - t1.tv_usec) / 1000000.0;
  double duration_vectoriel = (double)(t3.tv_sec - t2.tv_sec) +
                              (double)(t3.tv_usec - t2.tv_usec) / 1000000.0;
  double duration_multithread = (double)(t4.tv_sec - t3.tv_sec) +
                                (double)(t4.tv_usec - t3.tv_usec) / 1000000.0;
  double duration_multithread_vectoriel =
      (double)(t5.tv_sec - t4.tv_sec) +
      (double)(t5.tv_usec - t4.tv_usec) / 1000000.0;

  double speedup_avx = duration_scalar / duration_vectoriel;
  double speedup_mt = duration_scalar / duration_multithread;
  double speedup_mt_avx = duration_scalar / duration_multithread_vectoriel;

  printf("Dist: %f, Time: %f seconds\n", dist_scalar, duration_scalar);
  printf("Dist avx: %f, Time: %f seconds, Speedup: %.2fx\n", dist_vectoriel,
         duration_vectoriel, speedup_avx);
  printf("Multi thread Flex Dist gen: %f, Time: %f seconds, Speedup: %.2fx\n",
         dist_multithread, duration_multithread, speedup_mt);
  printf(
      "Multi thread Flex Dist avx gen: %f, Time: %f seconds, Speedup: %.2fx\n",
      dist_multithread_vectoriel, duration_multithread_vectoriel,
      speedup_mt_avx);

  // free memory
  free(u);
  free(v);
  return 0;
}
