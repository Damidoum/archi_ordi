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
    float ui = u[i];
    float vi = v[i];
    float squareroot =
        sqrtf((ui * ui + (vi * vi)) / (1 + (ui * vi) * (ui * vi)));
    // printf("ui: %f, vi: %f, squareroot: %f\n", ui, vi, squareroot);
    sum += squareroot;
    // printf("sum: %f\n", sum);
  }
  return sum;
}

// exercice 2
double dist_avx(float *u, float *v, int n) {
  __m256 vec_sum = _mm256_setzero_ps();
  __m256 one = _mm256_set1_ps(1.0);

  for (int i = 0; i < n - 7; i += 8) {
    __m256 vec_u = _mm256_load_ps(&u[i]);
    // printf("vec_u: %f %f %f %f\n", ((double *)&vec_u)[0], ((double
    // *)&vec_u)[1], ((double *)&vec_u)[2], ((double *)&vec_u)[3]);
    __m256 vec_v = _mm256_load_ps(&v[i]);
    // printf("vec_v: %f %f %f %f\n", ((double *)&vec_v)[0], ((double
    // *)&vec_v)[1], ((double *)&vec_v)[2], ((double *)&vec_v)[3]);
    __m256 vec_sqrt = _mm256_sqrt_ps(_mm256_div_ps(
        _mm256_add_ps(_mm256_mul_ps(vec_u, vec_u), _mm256_mul_ps(vec_v, vec_v)),
        _mm256_add_ps(one, _mm256_mul_ps(_mm256_mul_ps(vec_u, vec_v),
                                         _mm256_mul_ps(vec_u, vec_v)))));
    // printf("vec_sqrt: %f %f %f %f\n", ((double *)&vec_sqrt)[0], ((double
    // *)&vec_sqrt)[1], ((double *)&vec_sqrt)[2], ((double *)&vec_sqrt)[3]);
    vec_sum = _mm256_add_ps(vec_sum, vec_sqrt);
    // printf("vec_sum: %f %f %f %f\n", ((double *)&vec_sum)[0], ((double
    // *)&vec_sum)[1], ((double *)&vec_sum)[2], ((double *)&vec_sum)[3]);
  }
  // vec_sum = _mm256_hadd_ps(vec_sum, vec_sum);
  // vec_sum = _mm256_hadd_ps(vec_sum, vec_sum);
  // printf("vec_sum_final_after1_hadd: %f %f %f %f\n", ((double *)&vec_sum)[0],
  // ((double *)&vec_sum)[1], ((double *)&vec_sum)[2], ((double *)&vec_sum)[3]);
  // vec_sum = _mm256_add_pd(vec_sum, _mm256_permute4x64_pd(vec_sum,
  // 0b00011011)); printf("vec_sum_final: %f %f %f %f\n", ((double
  // *)&vec_sum)[0], ((double
  // *)&vec_sum)[1], ((double *)&vec_sum)[2], ((double *)&vec_sum)[3]);

  double result = 0.0;
  float temp[8];
  _mm256_storeu_ps(temp, vec_sum);
  for (int i = 0; i < 8; i++)
    result += (double)temp[i];

  /*
  for (int i = 0; i < 8; i++)
    result += temp[i];
  if (n % 8 != 0) {
    for (int i = n - (n % 8); i < n; i++) {
      float ui = u[i];
      float vi = v[i];
      double squareroot =
          sqrtf((ui * ui + (vi * vi)) / (1 + (ui * vi) * (ui * vi)));
      result += squareroot;
    }
  }
  */
  return result;
}

// exercice 3
double dist_avx_gen(float *u, float *v, int n) {
  __m256 vec_sum = _mm256_setzero_ps();
  __m256 one = _mm256_set1_ps(1.0);

  for (int i = 0; i < n - 7; i += 8) {
    __m256 vec_u = _mm256_loadu_ps(&u[i]);
    __m256 vec_v = _mm256_loadu_ps(&v[i]);
    __m256 vec_sqrt = _mm256_sqrt_ps(_mm256_div_ps(
        _mm256_add_ps(_mm256_mul_ps(vec_u, vec_u), _mm256_mul_ps(vec_v, vec_v)),
        _mm256_add_ps(one, _mm256_mul_ps(_mm256_mul_ps(vec_u, vec_v),
                                         _mm256_mul_ps(vec_u, vec_v)))));
    vec_sum = _mm256_add_ps(vec_sum, vec_sqrt);
  }

  // vec_sum = _mm256_hadd_pd(vec_sum, vec_sum);
  // vec_sum = _mm256_add_pd(vec_sum, _mm256_permute4x64_pd(vec_sum,
  // 0b00011011));
  double result = 0.0;
  float temp[8];
  _mm256_storeu_ps(temp, vec_sum);
  for (int i = 0; i < 8; i++)
    result += (double)temp[i];

  if (n % 8 != 0) {
    for (int i = n - (n % 8); i < n; i++) {
      float ui = u[i];
      float vi = v[i];
      double squareroot =
          sqrtf((ui * ui + (vi * vi)) / (1 + (ui * vi) * (ui * vi)));
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
    return dist_avx_gen(&u[a], &v[a], end - a);
  } else {
    fprintf(stderr, "Invalid mode: %d\n", mode);
    return -1.0;
  }
}

// exercice 5
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
  pthread_t threads[nb_threads];
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
    // printf("Creating thread %d for range [%d, %d)\n", i, thread_args[i].a,
    // thread_args[i].b);
    pthread_create(&threads[i], NULL, flex_dist_gen_vec,
                   (void *)&thread_args[i]);
  }

  /* Wait for all threads to complete */
  double sum = 0;
  for (int i = 0; i < nb_threads; i++) {
    pthread_join(threads[i], NULL);
    sum += results[i];
  }
  // pthread_exit(NULL);
  free(results);
  free(thread_args);
  return sum;
}

int main() {
#define N 1024 * 1024
  float *u = (float *)aligned_alloc(32, N * sizeof(float));
  float *v = (float *)aligned_alloc(32, N * sizeof(float));
  for (int i = 0; i < N; i++) {
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

  // printf("Dist: %f\n", dist(u, v, N));
  // printf("Dist avx: %f\n", dist_avx(u, v, N));
  // printf("Dist avx gen: %f\n", dist_avx_gen(u, v, N));
  // printf("Flex Dist gen : %f\n", flex_dist_gen(u, v, N, 0, N, 0));
  // printf("Flex Dist avx gen: %f\n", flex_dist_gen(u, v, N, 0, N, 1));
  // printf("Multi thread Flex Dist gen: %f\n", distPar(u, v, N, 2, 0));
  // printf("Multi thread Flex Dist avx gen: %f\n", distPar(u, v, N, 2, 1));
  free(u);
  free(v);
  return 0;
}
