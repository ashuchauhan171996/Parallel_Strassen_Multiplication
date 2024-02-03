#include<iostream>
#include<vector>
#include<limits.h>
#include<math.h>
#include<time.h>

#include<omp.h>


using namespace std;
#define MAX_THREADS     2000


// Standard algorithm of Matrix Multiplication
vector<vector<int>> standard_matrix_mul(vector<vector<int>> &A, vector<vector<int>> &B) {
    int n = A.size();
    vector<vector<int>> C (n, vector<int> (n, 0));
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            for (int k=0; k<n; k++) {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Strassen's Matrix Multiplication Additon
vector<vector<int>> add_mat_strassen(vector<vector<int>> P, vector<vector<int>> Q) {
    int n = P.size();
    int sign = 1;
    vector<vector<int>> R (n, vector<int> (n, 0));
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            R[i][j] = P[i][j] + (sign/abs(sign))*Q[i][j];
        }
    }
    return R;
}


// Strassen's Matrix Multiplication Subtraction
vector<vector<int>> subtract_mat_strassen(vector<vector<int>> P, vector<vector<int>> Q) {
    int n = P.size();
    int sign = -1;
    vector<vector<int>> R (n, vector<int> (n, 0));
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            R[i][j] = P[i][j] + (sign/abs(sign))*Q[i][j];
        }
    }
    return R;
}



// OpenMP Strassen's Matrix Muliplication
vector<vector<int>> strassen_matrix_mul(vector<vector<int>> A, vector<vector<int>> B, int s) {
    int sizeA = A.size();
    int sizeB = B.size();
    if (sizeA <= s) {
        return standard_matrix_mul(A, B);
    }
    vector<vector<int>> C (sizeA, vector<int>(sizeB, 0));
    int sizeA2 = sizeA/2;
    int sizeB2 = sizeB/2;

    // Allocate memories to equal sized matrix blocks A11, A12, A21, A22, B11, B12, B21, B22, C11, C12, C21, C22
    vector<vector<int>> A11(sizeA2, vector<int>(sizeA2, 0));
    vector<vector<int>> A12(sizeA2, vector<int>(sizeA2, 0));
    vector<vector<int>> A21(sizeA2, vector<int>(sizeA2, 0));
    vector<vector<int>> A22(sizeA2, vector<int>(sizeA2, 0));
    vector<vector<int>> B11(sizeB2, vector<int>(sizeB2, 0));
    vector<vector<int>> B12(sizeB2, vector<int>(sizeB2, 0));
    vector<vector<int>> B21(sizeB2, vector<int>(sizeB2, 0));
    vector<vector<int>> B22(sizeB2, vector<int>(sizeB2, 0));
    vector<vector<int>> C11(sizeA2, vector<int>(sizeB2, 0));
    vector<vector<int>> C12(sizeA2, vector<int>(sizeB2, 0));
    vector<vector<int>> C21(sizeA2, vector<int>(sizeB2, 0));
    vector<vector<int>> C22(sizeA2, vector<int>(sizeB2, 0));

    // Allocate memories to Strassen's algorithm matrix blocks M1, M2, M3, M4, M5, M6, M7
    vector<vector<int>> M1(sizeA2, vector<int>(sizeB2, 0));
    vector<vector<int>> M2(sizeA2, vector<int>(sizeB2, 0));
    vector<vector<int>> M3(sizeA2, vector<int>(sizeB2, 0));
    vector<vector<int>> M4(sizeA2, vector<int>(sizeB2, 0));
    vector<vector<int>> M5(sizeA2, vector<int>(sizeB2, 0));
    vector<vector<int>> M6(sizeA2, vector<int>(sizeB2, 0));
    vector<vector<int>> M7(sizeA2, vector<int>(sizeB2, 0));
    

    // Parallelizing the tasks using OpenMP
    #pragma omp parallel for
    for(int i = 0; i < sizeA2; i++)
    {
        #pragma omp parallel for
        for(int j = 0; j < sizeB2; j++)
        {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][sizeA2 + j];

            A21[i][j] = A[sizeA2 + i][j];
            A22[i][j] = A[sizeA2 + i][sizeA2 + j];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][sizeB2 + j];
            
            B21[i][j] = B[sizeB2 + i][j];
            B22[i][j] = B[sizeB2 + i][sizeB2 + j];
        }
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            { M1 = strassen_matrix_mul(add_mat_strassen(A11, A22), add_mat_strassen(B11, B22), s);}
            #pragma omp task
            { M2 = strassen_matrix_mul(add_mat_strassen(A21, A22), B11, s);}
            #pragma omp task
            { M3 = strassen_matrix_mul(A11, subtract_mat_strassen(B12, B22), s);}
            #pragma omp task
            { M4 = strassen_matrix_mul(A22, subtract_mat_strassen(B21, B11), s);}
            #pragma omp task
            { M5 = strassen_matrix_mul(add_mat_strassen(A11, A12), B22, s);}
            #pragma omp task
            { M6 = strassen_matrix_mul(subtract_mat_strassen(A21, A11), add_mat_strassen(B11, B12), s);}
            #pragma omp task
            { M7 = strassen_matrix_mul(subtract_mat_strassen(A12, A22), add_mat_strassen(B21, B22), s);}
            #pragma omp taskwait
        }
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            { C11 = add_mat_strassen(subtract_mat_strassen(add_mat_strassen(M1, M4), M5), M7);}
            #pragma omp task
            { C12 = add_mat_strassen(M3, M5);}
            #pragma omp task
            { C21 = add_mat_strassen(M2, M4);}
            #pragma omp task
            { C22 = add_mat_strassen(add_mat_strassen(subtract_mat_strassen(M1, M2), M3), M6);}
            #pragma omp taskwait
        }
    }

    // Calculating final product matrix
    #pragma omp parallel for
    for(int i = 0; i < sizeA2; i++)
    {
        #pragma omp parallel for
        for(int j = 0; j < sizeB2; j++)
        {
            C[i][j] = C11[i][j];
            C[i][sizeB2 + j] = C12[i][j];
            C[sizeA2 + i][j] = C21[i][j];
            C[sizeA2 + i][sizeB2 + j] = C22[i][j];
        }
    }

    return C;
}

// Initializing matrix 
void initialization_matrix(vector<vector<int>> &Mat) {
    for (auto i=Mat.begin(); i<Mat.end(); i++) {
        for(auto j=(*i).begin(); j<(*i).end(); j++) {
            *j = rand() % 100;
        }
    }
}

// calculate error
int check_matrices(vector<vector<int>> A, vector<vector<int>> B) {
    int sizeA = A.size();
    int sizeB = B.size();
    for(int i=0; i<sizeA; i++) {
        for(int j=0; j<sizeB; j++) {
            if (A[i][j] != B[i][j]) {
                return 1;
            }
        }
    }
    return 0;
}


// Main Function
int main(int argc, char *argv[]) {

    struct timespec start_time, stop_time, start_std, stop_std;
    int k, k_dash, num_threads, n, s;

    k=atoi(argv[1]);
    n= pow(2, k);
    k_dash = atoi(argv[2]);
    s = pow(2, k_dash);
    num_threads = atoi(argv[3]);;

     if (num_threads  > MAX_THREADS) {
        printf("Maximum number of threads allowed: %d.\n", MAX_THREADS);
        exit(0);
        }

     if(n<1 || n%2!=0){
		
		printf("\nEnter a  matrix dimension that is power of 2 only!\n\n");
		exit(0);
	
	}

    double time_strassen_matrix_mul;
    double time_standard_matrix_mul;

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    
    // Allocating A and B matrices 
    vector<vector<int>> A(n, vector<int>(n, 0));
    vector<vector<int>> B(n, vector<int>(n, 0));
    // Initializing A and B
    initialization_matrix(A);
    initialization_matrix(B);

    // Creating two C matrices for Straussen and Standard Algorithm
    vector<vector<int>> C_standard(n, vector<int>(n, 0));
    vector<vector<int>> C_strassen(n, vector<int>(n, 0));
     
    // Time Taken in Strassen algorithm
    clock_gettime(CLOCK_REALTIME, &start_time);
    C_strassen = strassen_matrix_mul(A, B, s);
    clock_gettime(CLOCK_REALTIME, &stop_time);
    time_strassen_matrix_mul = (stop_time.tv_sec-start_time.tv_sec) +0.000000001*(stop_time.tv_nsec-start_time.tv_nsec);
    
    // Time Taken in Standard algorithm
    clock_gettime(CLOCK_REALTIME, &start_std);
    C_standard = standard_matrix_mul(A, B);
    clock_gettime(CLOCK_REALTIME, &stop_std);
    time_standard_matrix_mul = (stop_std.tv_sec-start_std.tv_sec) +0.000000001*(stop_std.tv_nsec-start_std.tv_nsec);
    
    int error = check_matrices(C_strassen, C_standard);
    if (error != 0) { printf("Result of Strassen and standard multiplication are not same\n"); }

    // Printing results
    printf("k = %d, k\' = %d, matrix_size = %d x %d, threads = %d, error = %d, strassen_time (sec) = %4.6f, standard_time (sec) = %4.6f\n", k, k_dash, n, n, num_threads, error, time_strassen_matrix_mul, time_standard_matrix_mul);
    return 0;
}
