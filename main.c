#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Yahir Figueroa Vega

#define REPETITIONS 10  // Represent runs for getting average time

// This functions reads a file with n agens and m goods
void read_input_file(char *filename, int *n, int *m, double ***V) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    perror("Error opening file");
    exit(1);
  }

  fscanf(file, "%d %d", n, m);

  // This allocates the valuation matrix V
  *V = malloc((*n) * sizeof(double *));
  for (int i = 0; i < *n; i++) {
    // This allocates each row
    (*V)[i] = malloc((*m) * sizeof(double));
    for (int j = 0; j < *m; j++) {
      // Reads each valuation converting to double
      fscanf(file, "%lf", &(*V)[i][j]);
    }
  }
  fclose(file);
}

// Function that allocated goods ensuring EF1 it is based on round robin
void EF1_allocate_sequential(int n, int m, double **V, int *owner) {
  // Initializes all goods as unassigned, this reduces interference with
  // previous runs
  for (int g = 0; g < m; g++) {
    owner[g] = -1;
  }

  // Represent the number of remaining goods to allocate
  int remaining_items = m;
  // Turns of the agents
  int turn = 0;
  // Continue loop untill all goods are assigned
  while (remaining_items > 0) {
    // Current agent
    int i = turn % n;
    double max_value = -1.0;
    int best_good = -1;
    // Find the most valued good that is unassigned
    for (int g = 0; g < m; g++) {
      if (owner[g] == -1 && V[i][g] > max_value) {
        max_value = V[i][g];
        best_good = g;
      }
    }
    // Assign the best good found to the current agent
    if (best_good != -1) {
      owner[best_good] = i;
      remaining_items--;
    }
    // next turn of the agent
    turn++;
  }
  printf("EF1 Allocation completed.\n");
}

// This function checks if the allocation received is EF1 sequentially
int EF1_check_sequential(int n, int m, double **V, int *owner) {
  // Flag that says if the allocation is EF1
  int is_EF1 = 1;

  // Compare every pair of agents
  for (int i = 0; i < n && is_EF1; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        continue;
      }

      double vi_i = 0;
      double vi_j = 0;
      // Calculate the value that agent gives to himself and to the other agent
      // bundle
      for (int k = 0; k < m; k++) {
        if (owner[k] == i) {
          vi_i += V[i][k];
        }
        if (owner[k] == j) {
          vi_j += V[i][k];
        }
      }
      // If there is no envy continue
      if (vi_i >= vi_j) {
        continue;
      }

      // check if envy can be removed
      int ok = 0;

      for (int g = 0; g < m; g++) {
        if (owner[g] == j) {
          // value of j without good g
          double vi_j_minus_g = vi_j - V[i][g];
          // if ther eis no envy everything is ok
          if (vi_i >= vi_j_minus_g) {
            ok = 1;
            break;
          }
        }
      }
      // if there is still envy the allocation received is not EF1
      if (ok == 0) {
        is_EF1 = 0;
        break;
      }
    }
  }
  if (is_EF1 == 1) {
    printf("The allocation is EF1\n");
  } else {
    printf("The allocation is not EF1\n");
  }
  return is_EF1;
}

// This functions has the exact same logic as sequential but used openMP
// directives
int EF1_check_parallel(int n, int m, double **V, int *owner) {
  int is_EF1 = 1;

  // Uses pragma omp parallel for collpase to read every pair of agents at once
#pragma omp parallel for collapse(2)

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        continue;
      }

      double vi_i = 0.0;
      double vi_j = 0.0;

      for (int k = 0; k < m; k++) {
        if (owner[k] == i) {
          vi_i += V[i][k];
        }
        if (owner[k] == j) {
          vi_j += V[i][k];
        }
      }

      if (vi_i >= vi_j) {
        continue;
      }

      int ok = 0;

      for (int g = 0; g < m; g++) {
        if (owner[g] == j) {
          double vi_j_minus_g = vi_j - V[i][g];
          if (vi_i >= vi_j_minus_g) {
            ok = 1;
            break;
          }
        }
      }
      // Atomic safely marks that the allocation is not Ef1
      if (ok == 0) {
#pragma omp atomic
        is_EF1 &= 0;
      }
    }
  }
  if (is_EF1 == 1) {
    printf("The allocation is EF1\n");
  } else {
    printf("The allocation is not EF1\n");
  }
  return is_EF1;
  fflush(stdout);
}

// Functions that allocate goods trying a proportionally fair behavior
// sequentially
void PF_allocate_sequential(int n, int m, double **V, int *owner) {
  // Initialize auxiliary arrays
  double *total_value = calloc(n, sizeof(double));
  double *quota = calloc(n, sizeof(double));
  double *value_received = calloc(n, sizeof(double));

  // Computes the total valuation and proportional quota each agent should have
  for (int i = 0; i < n; i++) {
    for (int g = 0; g < m; g++) {
      total_value[i] += V[i][g];
    }
    quota[i] = total_value[i] / n;
  }
  // initialize all goods as unassigned
  for (int g = 0; g < m; g++) {
    owner[g] = -1;
  }

  for (int g = 0; g < m; g++) {
    // Find the agent with the lowest ratio of value received to quota
    int best_agent = -1;
    double min_remaining_quota = 1e9;
    for (int i = 0; i < n; i++) {
      double ratio;
      if (quota[i] > 0) {
        // calculates the ratio of the value received to his quota
        ratio = value_received[i] / quota[i];
      } else {
        ratio = 1;
      }
      // If the ratio is the lowest found update best agent
      if (ratio < min_remaining_quota) {
        min_remaining_quota = ratio;
        best_agent = i;
      }
    }
    // assigns the good to the best agent found
    owner[g] = best_agent;
    value_received[best_agent] += V[best_agent][g];
  }
  printf("PF Allocation completed.\n");
  free(total_value);
  free(quota);
  free(value_received);
}

// This implements the same logic as sequential but using openMP directives
void PF_allocate_parallel(int n, int m, double **V, int *owner) {
  double *total_value = calloc(n, sizeof(double));
  double *quota = calloc(n, sizeof(double));
  double *value_received = calloc(n, sizeof(double));

// Uses pragma omp parallel for to compute total values and quotas for eaxch
// agent in parallel
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int g = 0; g < m; g++) {
      total_value[i] += V[i][g];
    }
    quota[i] = total_value[i] / n;
  }

  for (int g = 0; g < m; g++) {
    owner[g] = -1;
  }

  for (int g = 0; g < m; g++) {
    int best_agent = -1;
    double min_remaining_quota = 1e9;
    for (int i = 0; i < n; i++) {
      double ratio;
      if (quota[i] > 0) {
        ratio = value_received[i] / quota[i];
      } else {
        ratio = 1;
      }
      if (ratio < min_remaining_quota) {
        min_remaining_quota = ratio;
        best_agent = i;
      }
    }
    owner[g] = best_agent;
    value_received[best_agent] += V[best_agent][g];
  }
  printf("PF Allocation completed.\n");
  free(total_value);
  free(quota);
  free(value_received);
}

// checks if the allocation received is PF sequentially
int PF_check_sequential(int n, int m, double **V, int *owner) {
  // initialize auxiliary arrays
  double *total_value = calloc(n, sizeof(double));
  double *value_received = calloc(n, sizeof(double));

  // Compute total valuations and received values
  for (int i = 0; i < n; i++) {
    for (int g = 0; g < m; g++) {
      total_value[i] += V[i][g];
    }
  }
  for (int g = 0; g < m; g++) {
    int a = owner[g];
    if (a >= 0) {
      // Sum the value recived by each agent
      value_received[a] += V[a][g];
    }
  }
  int is_PF = 1;
  for (int i = 0; i < n; i++) {
    // Check if each agent recived its respective quota
    double required_value = total_value[i] / n;
    if (value_received[i] < required_value) {
      // If agent didnt receive his quota it means allocation is not PF
      is_PF = 0;
      break;
    }
  }
  if (is_PF == 1) {
    printf("The allocation is PF\n");
  } else {
    printf("The allocation is not PF\n");
  }
  free(total_value);
  free(value_received);
  return is_PF;
}

// This function implements the same logic as sequential but using openMP
// directives
int PF_check_parallel(int n, int m, double **V, int *owner) {
  double *total_value = calloc(n, sizeof(double));
  double *value_received = calloc(n, sizeof(double));

// Parallelize calculaitons of total valuations
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int g = 0; g < m; g++) {
      total_value[i] += V[i][g];
    }
  }
  // safely paralelize calculation of values received
#pragma omp parallel for
  for (int g = 0; g < m; g++) {
    int a = owner[g];
    if (a >= 0) {
      // uses atomic tomake sure value received sum is correct
#pragma omp atomic
      value_received[a] += V[a][g];
    }
  }
  int is_PF = 1;
  for (int i = 0; i < n; i++) {
    double required_value = total_value[i] / n;
    if (value_received[i] < required_value) {
      is_PF = 0;
      break;
    }
  }
  if (is_PF == 1) {
    printf("The allocation is PF\n");
  } else {
    printf("The allocation is not PF\n");
  }
  free(total_value);
  free(value_received);
  return is_PF;
}

void test_EF1_check_yes() {
  // Number of agents and goods
  int n = 2, m = 2;
  double **V = malloc(n * sizeof(double *));
  for (int i = 0; i < n; i++) {
    V[i] = malloc(m * sizeof(double));
  }
  int owner[] = {1, 0};

  // Valuations, agent 0 values good 0 at 9 and good 1 at 1 agent 1 values both
  // goods at 0
  V[0][0] = 9;
  V[0][1] = 1;
  V[1][0] = 0;
  V[1][1] = 0;

  // If the allocations result in EF1 the test passes
  printf("Running EF1_check:\n");
  int result1 = EF1_check_sequential(n, m, V, owner);
  int result2 = EF1_check_parallel(n, m, V, owner);
  if (result1 == 1 && result2 == 1) {
    printf("EF1_check test passed: Allocation is EF1 as expected.\n");
  } else {
    printf("EF1_check test failed: Allocation is not EF1, but it should be.\n");
  }
}

// Same logic as previous test
void test_EF1_check_no() {
  int n = 2, m = 2;
  double **V = malloc(n * sizeof(double *));
  for (int i = 0; i < n; i++) {
    V[i] = malloc(m * sizeof(double));
  }

  // Generate envy situation that cant be resolved making the allocation not Ef1
  int owner[] = {1, 1};

  V[0][0] = 10;
  V[0][1] = 10;
  V[1][0] = 0;
  V[1][1] = 0;

  printf("Running EF1_check:\n");
  int result1 = EF1_check_sequential(n, m, V, owner);
  int result2 = EF1_check_parallel(n, m, V, owner);

  if (result1 == 0 && result2 == 0) {
    printf("EF1_check test passed: Allocation is not EF1 as expected.\n");
  } else {
    printf("EF1_check test failed: Allocation is EF1, but it should not be.\n");
  }
}

// Same logic as previous tests
void test_PF_check_yes() {
  int n = 2, m = 2;
  double **V = malloc(n * sizeof(double *));
  for (int i = 0; i < n; i++) {
    V[i] = malloc(m * sizeof(double));
  }
  int owner[] = {0, 1};

  V[0][0] = 10;
  V[0][1] = 10;
  V[1][0] = 10;
  V[1][1] = 10;

  printf("Running PF_check:\n");
  int result1 = PF_check_sequential(n, m, V, owner);
  int result2 = PF_check_parallel(n, m, V, owner);

  if (result1 == 1 && result2 == 1) {
    printf("PF_check test passed: Allocation is PF as expected.\n");
  } else {
    printf("PF_check test failed: Allocation is not PF, but it should be.\n");
  }
}

// Same logic as previus tests
void test_PF_check_no() {
  int n = 2, m = 2;
  double **V = malloc(n * sizeof(double *));
  for (int i = 0; i < n; i++) {
    V[i] = malloc(m * sizeof(double));
  }
  int owner[] = {0, 0};

  V[0][0] = 10;
  V[0][1] = 10;
  V[1][0] = 10;
  V[1][1] = 10;

  printf("Running PF_check:\n");
  int result1 = PF_check_sequential(n, m, V, owner);
  int result2 = PF_check_parallel(n, m, V, owner);

  if (result1 == 0 && result2 == 0) {
    printf("PF_check test passed: Allocation is not PF as expected.\n");
  } else {
    printf("PF_check test failed: Allocation is PF, but it should not be.\n");
  }
}

void test_PF_allocate() {
  int n = 2, m = 2;
  double **V = malloc(n * sizeof(double *));
  for (int i = 0; i < n; i++) {
    V[i] = malloc(m * sizeof(double));
  }
  int owner_seq[2];
  int owner_par[2];

  V[0][0] = 10;
  V[0][1] = 1;
  V[1][0] = 1;
  V[1][1] = 10;

  PF_allocate_sequential(n, m, V, owner_seq);
  PF_allocate_parallel(n, m, V, owner_par);
  printf("Running PF_Allocation_check:\n");
  int result1 = PF_check_sequential(n, m, V, owner_seq);
  int result2 = PF_check_parallel(n, m, V, owner_par);

  if (result1 == 1 && result2 == 1) {
    printf("PF_allocate test passed: Both allocations are PF as expected.\n");
  } else {
    printf("PF_allocate test failed: One or both allocations are not PF.\n");
  }
}

int main(int argc, char *argv[]) {
  // prints threads used
  int threads = omp_get_max_threads();
  printf("Using %d threads\n", threads);
  srand(1234);
  if (argc < 2) {
    printf("Usage invalid\n");
    return 1;
  }

  int n, m;
  double **V;

  // calls read matrix functions
  read_input_file(argv[1], &n, &m, &V);
  // Calls the tests
  test_EF1_check_yes();
  test_EF1_check_no();

  test_PF_check_yes();
  test_PF_check_no();

  test_PF_allocate();

  printf("Matrix size: n=%d, m=%d\n", n, m);

  int *owner_sequential = malloc(sizeof(int) * m);
  int *owner_parallel = malloc(sizeof(int) * m);
  // The following sections call their functions and run 10 times calculating
  // their averages The average calculated are speedups and execution times
  //  EF1 ALLOCATION
  EF1_allocate_sequential(n, m, V, owner_sequential);
  for (int i = 0; i < m; i++) {
    owner_parallel[i] = owner_sequential[i];
  }

  double ef1_sequential_time = 0.0;
  double ef1_parallel_time = 0.0;
  for (int rep = 0; rep < REPETITIONS; rep++) {
    double t0_seq = omp_get_wtime();
    EF1_check_sequential(n, m, V, owner_sequential);
    double t1_seq = omp_get_wtime();
    ef1_sequential_time += (t1_seq - t0_seq);

    printf("EF1 sequential check elapsed time: %.6f seconds in run %d\n",
           (t1_seq - t0_seq), rep + 1);
  }

  for (int rep = 0; rep < REPETITIONS; rep++) {
    double t0_parallel = omp_get_wtime();
    EF1_check_parallel(n, m, V, owner_parallel);
    double t1_parallel = omp_get_wtime();
    ef1_parallel_time += (t1_parallel - t0_parallel);

    printf("EF1 parallel check elapsed time: %.6f seconds in run %d\n",
           (t1_parallel - t0_parallel), rep + 1);
  }

  printf("\nEF1 Check Sequential Average Time over %d runs: %.6f seconds\n",
         REPETITIONS, ef1_sequential_time / REPETITIONS);

  printf("\nEF1 Check Parallel Average Time over %d runs: %.6f seconds\n",
         REPETITIONS, ef1_parallel_time / REPETITIONS);

  printf("\nEF1 Check Speedup: %.2f\n",
         ef1_sequential_time / ef1_parallel_time);

  // Proportional Fairness Allocation

  double pf_allocate_sequential_time = 0.0;
  double pf_allocate_parallel_time = 0.0;

  for (int rep = 0; rep < REPETITIONS; rep++) {
    double t0_seq = omp_get_wtime();
    PF_allocate_sequential(n, m, V, owner_sequential);
    double t1_seq = omp_get_wtime();
    pf_allocate_sequential_time += (t1_seq - t0_seq);
    printf("PF sequential allocation elapsed time: %.6f seconds in run %d\n",
           (t1_seq - t0_seq), rep + 1);
  }

  for (int rep = 0; rep < REPETITIONS; rep++) {
    double t0_parallel = omp_get_wtime();
    PF_allocate_parallel(n, m, V, owner_parallel);
    double t1_parallel = omp_get_wtime();
    pf_allocate_parallel_time += (t1_parallel - t0_parallel);

    printf("PF parallel allocation elapsed time: %.6f seconds in run %d\n",
           (t1_parallel - t0_parallel), rep + 1);
  }

  printf("\nPF Allocation Sequential Average Time over %d runs: %.6f seconds\n",
         REPETITIONS, pf_allocate_sequential_time / REPETITIONS);
  printf("\nPF Allocation Parallel Average Time over %d runs: %.6f seconds\n",
         REPETITIONS, pf_allocate_parallel_time / REPETITIONS);
  printf("\nPF Allocation Speedup: %.2f\n",
         pf_allocate_sequential_time / pf_allocate_parallel_time);

  for (int g = 0; g < m; g++) {
    if (owner_parallel[g] != owner_sequential[g]) {
      printf("Mismatch in allocation at good %d: seq=%d, par=%d\n", g,
             owner_sequential[g], owner_parallel[g]);
      break;
    }
  }

  double pf_check_sequential_time = 0.0;
  double pf_check_parallel_time = 0.0;

  for (int rep = 0; rep < REPETITIONS; rep++) {
    double t0_seq = omp_get_wtime();
    PF_check_sequential(n, m, V, owner_sequential);
    double t1_seq = omp_get_wtime();
    pf_check_sequential_time += (t1_seq - t0_seq);

    printf("PF sequential check elapsed time: %.6f seconds in run %d\n",
           (t1_seq - t0_seq), rep + 1);
  }
  for (int rep = 0; rep < REPETITIONS; rep++) {
    double t0_parallel = omp_get_wtime();
    PF_check_parallel(n, m, V, owner_parallel);
    double t1_parallel = omp_get_wtime();
    pf_check_parallel_time += (t1_parallel - t0_parallel);

    printf("PF parallel check elapsed time: %.6f seconds in run %d\n",
           (t1_parallel - t0_parallel), rep + 1);
  }

  printf("\nPF Check Sequential Average Time over %d runs: %.6f seconds\n",
         REPETITIONS, pf_check_sequential_time / REPETITIONS);
  printf("\nPF Check Parallel Average Time over %d runs: %.6f seconds\n",
         REPETITIONS, pf_check_parallel_time / REPETITIONS);
  printf("\nPF Check Speedup: %.2f\n",
         pf_check_sequential_time / pf_check_parallel_time);

  for (int i = 0; i < n; i++) {
    free(V[i]);
  }
  free(V);

  free(owner_sequential);
  free(owner_parallel);
  return 0;
}
