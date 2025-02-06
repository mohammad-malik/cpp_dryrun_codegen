// IMPORTANT: compile with -pthread flag, e.g.:
//   g++ -pthread -o matrix_ops matrix_ops.cpp

#include <iostream>
#include <pthread.h>
#include <cstring>    // for memset if needed

using namespace std;

// Global constants
const int MATRIX_SIZE = 1024;   // full matrix dimensions
const int DET_SIZE = 64;        // size of submatrix for determinant computation
const int THREAD_COUNT = 6;

// -----------------------
// Utility routines for 2D matrix allocation and deallocation
// -----------------------
double** allocateMatrix(int rows, int cols)
{
    double **mat = new double*[rows];
    for (int i = 0; i < rows; i++){
        mat[i] = new double[cols];
    }
    return mat;
}

void deallocateMatrix(double **mat, int rows)
{
    for (int i = 0; i < rows; i++){
        delete [] mat[i];
    }
    delete [] mat;
}

// -----------------------
// A simple approximation to the natural logarithm
// (using the Mercator series for ln((1+y)/(1-y)) with y=(x-1)/(x+1))
// Note: for x > 0 only and converges fast when x is near 1.
// (In practice, one would use math library functions.)
// -----------------------
double my_log(double x)
{
    if (x <= 0) {
        // for simplicity, we just return a large negative number for non-positive input
        return -1e9;
    }
    double y = (x - 1) / (x + 1);
    double result = 0.0;
    double term = y;
    const int iterations = 20;  // more iterations -> better accuracy
    for (int n = 0; n < iterations; n++){
         result += term / (2*n + 1);
         term *= (y * y);
    }
    return 2 * result;
}

// -----------------------
// Global matrices (original input and buffers for multi-threaded results)
// -----------------------
double **g_originalMatrix = nullptr;   // 1024x1024 original matrix
double **g_detMatrix = nullptr;          // working copy for determinant (size DET_SIZE)
double **g_transMatrix = nullptr;        // output for matrix transposition
double **g_logMatrix = nullptr;          // output for log transformation

// Also store sequential (gold) results for checking:
double g_seqDet = 0.0;
double **g_seqTrans = nullptr;
double **g_seqLog   = nullptr;

// -----------------------
// 1. MULTI-THREADED DETERMINANT COMPUTATION (using Block distribution)
//    We use a parallel Gaussian elimination on the DET_SIZE×DET_SIZE submatrix.
//    (The algorithm is similar to sequential elimination but the elimination of rows 
//     below the pivot row is distributed among THREAD_COUNT threads using block distribution.)
// -----------------------

// Global variables for the determinant threads:
pthread_barrier_t det_barrier;  // barrier to synchronize threads during elimination
volatile double g_detProduct = 1.0;  // global product of diagonal entries (determinant)
int g_det_n = DET_SIZE;         // size of the submatrix

// In the parallel elimination algorithm, each thread is assigned a contiguous block of rows 
// (from pivot k+1 to end) for each elimination step.
struct DetThreadData {
    int tid; // thread id [0..THREAD_COUNT-1]
};

void* determinantWorker(void* arg)
{
    DetThreadData* data = (DetThreadData*) arg;
    int tid = data->tid;
    int n = g_det_n;
    
    // For each pivot row k, update rows i = k+1...n-1 in blocks (using block distribution).
    for (int k = 0; k < n; k++) {
        // Wait so that all threads have finished previous elimination steps.
        pthread_barrier_wait(&det_barrier);
        
        // Let thread 0 update the determinant product (this is done only once per iteration).
        if (tid == 0) {
            g_detProduct *= g_detMatrix[k][k];
        }
        
        // Wait until pivot multiplication is done.
        pthread_barrier_wait(&det_barrier);
        
        // Determine the range of rows to update:
        int totalRows = n - (k + 1);
        int rowsPerThread = (totalRows + THREAD_COUNT - 1) / THREAD_COUNT; // ceiling division
        int start = k + 1 + tid * rowsPerThread;
        int end = start + rowsPerThread;
        if (end > n) end = n;
        
        // Update the assigned rows: eliminate column k using pivot row k.
        for (int i = start; i < end; i++) {
            double factor = g_detMatrix[i][k] / g_detMatrix[k][k];
            // update row i for columns k+1 to n-1
            for (int j = k + 1; j < n; j++){
                g_detMatrix[i][j] -= factor * g_detMatrix[k][j];
            }
            // set the lower element to 0
            g_detMatrix[i][k] = 0.0;
        }
        // Synchronize so that all threads finish elimination for this pivot.
        pthread_barrier_wait(&det_barrier);
    }
    pthread_exit(nullptr);
}

// Sequential determinant computation using Gaussian elimination on a copy of the submatrix.
double sequentialDeterminant(double **mat, int n)
{
    // create a copy so that we don’t modify the original
    double **temp = allocateMatrix(n, n);
    for (int i = 0; i < n; i++){
       for (int j = 0; j < n; j++){
          temp[i][j] = mat[i][j];
       }
    }
    double det = 1.0;
    for (int k = 0; k < n; k++){
        det *= temp[k][k];
        for (int i = k+1; i < n; i++){
            double factor = temp[i][k] / temp[k][k];
            for (int j = k+1; j < n; j++){
                temp[i][j] -= factor * temp[k][j];
            }
            temp[i][k] = 0.0;
        }
    }
    deallocateMatrix(temp, n);
    return det;
}

// Function to run the multi-threaded determinant computation.
double multiThreadedDeterminant()
{
    // initialize barrier (3 barriers per pivot step, but we reuse one barrier)
    pthread_barrier_init(&det_barrier, nullptr, THREAD_COUNT);
    
    pthread_t threads[THREAD_COUNT];
    DetThreadData threadData[THREAD_COUNT];
    
    // Launch THREAD_COUNT threads.
    for (int i = 0; i < THREAD_COUNT; i++){
        threadData[i].tid = i;
        pthread_create(&threads[i], nullptr, determinantWorker, (void*) &threadData[i]);
    }
    
    // Wait for threads to finish.
    for (int i = 0; i < THREAD_COUNT; i++){
        pthread_join(threads[i], nullptr);
    }
    
    pthread_barrier_destroy(&det_barrier);
    
    return g_detProduct;
}

// -----------------------
// 2. MULTI-THREADED MATRIX TRANSPOSITION (using row-wise cyclic distribution)
//    Each thread handles every THREAD_COUNT–th row (i.e. rows i with i mod THREAD_COUNT == tid)
// -----------------------
struct TransThreadData {
    int tid;
};

void* transposeWorker(void* arg)
{
    TransThreadData* data = (TransThreadData*) arg;
    int tid = data->tid;
    for (int i = tid; i < MATRIX_SIZE; i += THREAD_COUNT) {
        for (int j = 0; j < MATRIX_SIZE; j++){
            // use a separate output buffer (g_transMatrix)
            g_transMatrix[j][i] = g_originalMatrix[i][j];
        }
    }
    pthread_exit(nullptr);
}

// Sequential transpose computation.
void sequentialTranspose(double **in, double **out, int n)
{
    for (int i = 0; i < n; i++){
       for (int j = 0; j < n; j++){
           out[j][i] = in[i][j];
       }
    }
}

// -----------------------
// 3. MULTI-THREADED ELEMENT–WISE LOG TRANSFORMATION (using row and column cyclic distribution)
//    Here each thread processes those elements whose linear index modulo THREAD_COUNT equals its tid.
// -----------------------
struct LogThreadData {
    int tid;
};

void* logWorker(void* arg)
{
    LogThreadData* data = (LogThreadData*) arg;
    int tid = data->tid;
    int totalElements = MATRIX_SIZE * MATRIX_SIZE;
    // Loop over all elements in row-major order; each thread picks its share.
    for (int idx = tid; idx < totalElements; idx += THREAD_COUNT) {
        int i = idx / MATRIX_SIZE;
        int j = idx % MATRIX_SIZE;
        // Use a separate buffer g_logMatrix for storing the transformation.
        g_logMatrix[i][j] = my_log(g_originalMatrix[i][j]);
    }
    pthread_exit(nullptr);
}

// Sequential log transformation.
void sequentialLogTransformation(double **in, double **out, int n)
{
    for (int i = 0; i < n; i++){
       for (int j = 0; j < n; j++){
           out[i][j] = my_log(in[i][j]);
       }
    }
}

// -----------------------
// Function to check that multi-threaded results match sequential ones.
// -----------------------
bool CorrectOutputCheck(double seqDet, double mtDet,
                        double **seqTrans, double **mtTrans,
                        double **seqLog, double **mtLog,
                        int n_trans, int n_log)
{
    // Check determinant (allow small numerical tolerance)
    if (fabs(seqDet - mtDet) > 1e-6) {
        cout << "Determinant mismatch: sequential = " << seqDet << ", multi-threaded = " << mtDet << endl;
        return false;
    }
    
    // Check transposition (for 1024x1024 matrix)
    for (int i = 0; i < n_trans; i++){
        for (int j = 0; j < n_trans; j++){
            if (fabs(seqTrans[i][j] - mtTrans[i][j]) > 1e-6) {
                cout << "Transpose mismatch at (" << i << ", " << j << "): "
                     << seqTrans[i][j] << " vs " << mtTrans[i][j] << endl;
                return false;
            }
        }
    }
    
    // Check log transformation (for 1024x1024 matrix)
    for (int i = 0; i < n_log; i++){
        for (int j = 0; j < n_log; j++){
            if (fabs(seqLog[i][j] - mtLog[i][j]) > 1e-6) {
                cout << "Log transform mismatch at (" << i << ", " << j << "): "
                     << seqLog[i][j] << " vs " << mtLog[i][j] << endl;
                return false;
            }
        }
    }
    return true;
}

// -----------------------
// MAIN: Allocate matrices, initialize them, run sequential and multi-threaded computations,
//       and then check correctness.
// -----------------------
int main()
{
    // Allocate the original matrix and initialize it.
    g_originalMatrix = allocateMatrix(MATRIX_SIZE, MATRIX_SIZE);
    // For reproducibility, fill the matrix with positive values (e.g. 1.0 + i + j/1000)
    for (int i = 0; i < MATRIX_SIZE; i++){
       for (int j = 0; j < MATRIX_SIZE; j++){
           g_originalMatrix[i][j] = 1.0 + i + j/1000.0;
       }
    }
    
    // ----- 1. Determinant computation on a DET_SIZE×DET_SIZE submatrix -----
    // Create a working copy for multi-threaded determinant computation.
    g_detMatrix = allocateMatrix(DET_SIZE, DET_SIZE);
    for (int i = 0; i < DET_SIZE; i++){
       for (int j = 0; j < DET_SIZE; j++){
           // take the top-left DET_SIZE×DET_SIZE submatrix
           g_detMatrix[i][j] = g_originalMatrix[i][j];
       }
    }
    
    // Compute sequential determinant.
    g_seqDet = sequentialDeterminant(g_detMatrix, DET_SIZE);
    
    // Reset working copy for multi-threaded version:
    for (int i = 0; i < DET_SIZE; i++){
       for (int j = 0; j < DET_SIZE; j++){
           g_detMatrix[i][j] = g_originalMatrix[i][j];
       }
    }
    // Reset the global product
    g_detProduct = 1.0;
    
    // Launch multi-threaded determinant computation.
    double mtDet = multiThreadedDeterminant();
    
    // ----- 2. Matrix Transposition (full 1024×1024) -----
    // Allocate output buffers.
    g_transMatrix = allocateMatrix(MATRIX_SIZE, MATRIX_SIZE);
    g_seqTrans   = allocateMatrix(MATRIX_SIZE, MATRIX_SIZE);
    
    // Compute sequential transpose.
    sequentialTranspose(g_originalMatrix, g_seqTrans, MATRIX_SIZE);
    
    // Launch THREAD_COUNT threads for transposition.
    pthread_t transThreads[THREAD_COUNT];
    TransThreadData transData[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++){
         transData[i].tid = i;
         pthread_create(&transThreads[i], nullptr, transposeWorker, (void*) &transData[i]);
    }
    for (int i = 0; i < THREAD_COUNT; i++){
         pthread_join(transThreads[i], nullptr);
    }
    
    // ----- 3. Element–wise Logarithm Transformation (full 1024×1024) -----
    g_logMatrix = allocateMatrix(MATRIX_SIZE, MATRIX_SIZE);
    g_seqLog    = allocateMatrix(MATRIX_SIZE, MATRIX_SIZE);
    
    // Compute sequential log transformation.
    sequentialLogTransformation(g_originalMatrix, g_seqLog, MATRIX_SIZE);
    
    // Launch THREAD_COUNT threads for log transformation.
    pthread_t logThreads[THREAD_COUNT];
    LogThreadData logData[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++){
         logData[i].tid = i;
         pthread_create(&logThreads[i], nullptr, logWorker, (void*) &logData[i]);
    }
    for (int i = 0; i < THREAD_COUNT; i++){
         pthread_join(logThreads[i], nullptr);
    }
    
    // ----- Correctness Check -----
    bool ok = CorrectOutputCheck(g_seqDet, mtDet, g_seqTrans, g_transMatrix, g_seqLog, g_logMatrix,
                                 MATRIX_SIZE, MATRIX_SIZE);
    if (ok) {
        cout << "All multi-threaded results match the sequential ones." << endl;
    } else {
        cout << "There is a mismatch in the results!" << endl;
    }
    
    // Optionally, print the computed determinant:
    cout << "Determinant (submatrix " << DET_SIZE << "x" << DET_SIZE << "): " << mtDet << endl;
    
    // Deallocate all matrices.
    deallocateMatrix(g_originalMatrix, MATRIX_SIZE);
    deallocateMatrix(g_detMatrix, DET_SIZE);
    deallocateMatrix(g_transMatrix, MATRIX_SIZE);
    deallocateMatrix(g_logMatrix, MATRIX_SIZE);
    deallocateMatrix(g_seqTrans, MATRIX_SIZE);
    deallocateMatrix(g_seqLog, MATRIX_SIZE);
    
    return 0;
}