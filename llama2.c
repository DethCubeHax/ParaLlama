/*
PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME: llama2_3035835571.c
* NAME: ISLAM Nafis Ul
* UID : 3035835571
* Development Platform: Ubuntu 20.04 Docker Container
* Remark: The whole assignment
* How to compile: (gcc -o llama2_3035835571 llama2_3035835571.c utilities.c -O2 -pthread -lm)
*/
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

// YOUR CODE STARTS HERE

// Addtional Header File Here
#include <pthread.h>

// Define a structure to hold thread-related data
typedef struct {
    int start, end;  // Defines the range of rows this thread will handle
    float* out;      // A pointer to the resultant vector
    float* vec;      // A pointer to the input vector
    float* mat;      // A pointer to the input matrix
    int col;         // The column count of the matrix
    int execute;     // A flag to signal the thread to start computing
    int terminate;   // A flag to signal the thread to terminate
    pthread_mutex_t lock;  // Mutex for thread synchronization
    pthread_cond_t cond;   // Condition variable for thread synchronization
} ThreadData;

// Variables to track resource usage for main thread and child threads
struct rusage main_usage;  // To track usage stats of the main thread
pthread_t *threads;        // Array of thread identifiers
ThreadData *thread_data;   // Array of thread data
int thr_count;             // Number of threads
struct rusage* thread_usage;  // Array to track usage stats of each thread



// Function that each thread will execute
void *thr_func(void *arg) {                     
    ThreadData *data = (ThreadData*)arg;        // Cast the void* argument to a ThreadData* 
    for(;;) {                                   // Start an infinite loop that only ends when signaled to terminate
        pthread_mutex_lock(&data->lock);         // Lock the mutex to ensure thread-safe access to shared data
        // If the thread is not signaled to execute or terminate, wait on the condition variable. 
        // This releases the mutex and makes the thread sleep until signaled
        while(!data->execute && !data->terminate)
            pthread_cond_wait(&data->cond, &data->lock);
        if(data->terminate) {                   // If signaled to terminate, unlock the mutex and break from loop
            pthread_mutex_unlock(&data->lock);
            break;
        }
        pthread_mutex_unlock(&data->lock);       // Unlock the mutex as the thread is about to compute
        // Compute the part of the matrix-vector multiplication assigned to this thread
        for(int i=data->start; i<data->end; i++) {
            data->out[i] = 0;
            for(int j=0; j<data->col; j++)
                data->out[i] += data->mat[i*data->col+j] * data->vec[j];
        }
        // Update the thread's usage stats
        getrusage(RUSAGE_SELF, &thread_usage[data->start / (data->end - data->start)]);
        pthread_mutex_lock(&data->lock);          // Lock the mutex again as the thread is about to modify shared data
        data->execute = 0;                        // Set the execute flag to false as the thread has finished computing
        pthread_cond_signal(&data->cond);         // Signal the condition variable in case the main thread is waiting  
        pthread_mutex_unlock(&data->lock);        // Unlock the mutex
    }
    pthread_exit(NULL);                           // Exit the thread
}



// Function to initialize the data structures and threads needed for the matrix-vector multiplication
int init_mat_vec_mul(int n) {
    thr_count = n;                                                         // Set the number of threads

    threads = malloc(sizeof(pthread_t) * thr_count);                       // Allocate memory for thread identifiers
    thread_data = malloc(sizeof(ThreadData) * thr_count);                  // Allocate memory for thread data
    thread_usage = malloc(sizeof(struct rusage) * thr_count);              // Allocate memory for thread usage stats

    if(!threads || !thread_data || !thread_usage)                          // Check if memory allocation was successful
        return -1;

    for(int i=0; i<thr_count; i++) {                                       // Initialize each thread
        pthread_mutex_init(&thread_data[i].lock, NULL);                    // Initialize the mutex for the thread
        pthread_cond_init(&thread_data[i].cond, NULL);                     // Initialize the condition variable for the thread
        thread_data[i].execute = 0;                                        // Initialize the execution flag to false
        thread_data[i].terminate = 0;                                      // Initialize the termination flag to false

        // Create the thread and make it start executing thr_func, passing the address of its ThreadData as argument
        pthread_create(&threads[i], NULL, thr_func, (void*)&thread_data[i]);
    }
    return 0;                                                              // Return 0 to indicate successful initialization
}



// Function to perform the actual matrix-vector multiplication
void mat_vec_mul(float* out, float* vec, float* mat, int col, int row) {
    int rows_per_thread = row / thr_count;                                 // Compute the number of rows per thread

    for(int i=0; i<thr_count; i++) {                                       // For each thread
        thread_data[i].start = i * rows_per_thread;                        // Compute the start index for this thread
        thread_data[i].end = (i < thr_count-1) ?                           // Compute the end index for this thread
            ((i+1) * rows_per_thread) : row;                               
        thread_data[i].out = out;                                          // Set the output vector
        thread_data[i].vec = vec;                                          // Set the input vector
        thread_data[i].mat = mat;                                          // Set the input matrix
        thread_data[i].col = col;                                          // Set the number of columns

        pthread_mutex_lock(&thread_data[i].lock);                          // Lock the mutex for safe access to shared data
        thread_data[i].execute = 1;                                        // Set the execute flag to true
        pthread_cond_signal(&thread_data[i].cond);                         // Signal the condition variable
        pthread_mutex_unlock(&thread_data[i].lock);                        // Unlock the mutex
    }

    for(int i=0; i<thr_count; i++) {                                       // For each thread
        pthread_mutex_lock(&thread_data[i].lock);                          // Lock the mutex for safe access to shared data
        while(thread_data[i].execute)                                      // Wait for the thread to finish executing
            pthread_cond_wait(&thread_data[i].cond, &thread_data[i].lock);
        pthread_mutex_unlock(&thread_data[i].lock);                        // Unlock the mutex
    }
}


// Function to wrap up the matrix-vector multiplication
int close_mat_vec_mul() {
    getrusage(RUSAGE_SELF, &main_usage);                                   // Get the usage stats for the main thread

    // Signal termination to all threads
    for(int i=0; i<thr_count; i++) {
        pthread_mutex_lock(&thread_data[i].lock);                          // Lock the mutex for safe access to shared data
        thread_data[i].terminate = 1;                                      // Set the termination flag to true
        pthread_cond_signal(&thread_data[i].cond);                         // Signal the condition variable
        pthread_mutex_unlock(&thread_data[i].lock);                        // Unlock the mutex
    }

    // Wait for the threads to terminate
    for(int i=0; i<thr_count; i++) {
        pthread_join(threads[i], NULL);                                    // Wait for the thread to terminate
    }

    for (int i = 0; i < thr_count; i++) {                                  // For each thread
        // Print the usage stats for the thread
        printf("Thread %d has completed - user: %ld.%06lds, system: %ld.%06lds\n", i,
               thread_usage[i].ru_utime.tv_sec, thread_usage[i].ru_utime.tv_usec,
               thread_usage[i].ru_stime.tv_sec, thread_usage[i].ru_stime.tv_usec);
    }

    // Print the usage stats for the main thread
    printf("main thread - user: %ld.%06lds, system: %ld.%06lds\n",
           main_usage.ru_utime.tv_sec, main_usage.ru_utime.tv_usec,
           main_usage.ru_stime.tv_sec, main_usage.ru_stime.tv_usec);

    for(int i=0; i<thr_count; i++) {                                       // For each thread
        pthread_mutex_destroy(&thread_data[i].lock);                       // Destroy the mutex
        pthread_cond_destroy(&thread_data[i].cond);                        // Destroy the condition variable
    }

    free(threads);                                                         // Free the memory allocated for the thread identifiers
    free(thread_data);                                                     // Free the memory allocated for the thread data
    free(thread_usage);                                                    // Free the memory allocated for the thread usage stats

    return 0;                                                              // Return 0 to indicate successful termination
}

// YOUR CODE ENDS HERE

int transformer(int token, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w) {
    
    // a few convenience variables
    int dim = p->dim, hidden_dim =  p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim*sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l*dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);
            
            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }
    
        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }
    
    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q=0; q<p->vocab_size; q++) { s->logits[q] /= 0.9f; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char* argv[]) {

    unsigned int seed;
    int thr_count;

    if (argc == 3) {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    } else {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    init_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1) { return 1; }

    // load tokenizer
    char** vocab = malloc(config.vocab_size * sizeof(char*));
    if (load_tokenizer(vocab, config.vocab_size) == 1) { return 1; }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);
    
    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len) {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end-start)/1000, config.seq_len / (double)(end-start)*1000);

    // cleanup
    close_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}