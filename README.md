# Parallel Matrix-Vector Multiplication in C

This project implements a parallel matrix-vector multiplication algorithm in C using the POSIX threads (pthreads) library. The code is designed to efficiently distribute the computation across multiple threads and provides a simple interface for performing matrix-vector multiplication.

## Features

- Parallel computation of matrix-vector multiplication using pthreads
- Efficient distribution of workload among threads
- Thread synchronization using mutexes and condition variables
- Tracking of resource usage for main thread and child threads
- Simple interface for initializing, performing, and closing the matrix-vector multiplication

## Usage

1. Compile the code using the following command:
   ```
   gcc -o llama2 llama2.c utilities.c -O2 -pthread -lm
   ```

2. Run the compiled executable with the desired number of threads:
   ```
   ./llama2 <seed> <thr_count>
   ```
   Replace `<seed>` with a seed value for random number generation and `<thr_count>` with the desired number of threads to use for the computation.

## Code Structure

The code is organized into the following main functions:

- `thr_func`: The function executed by each thread. It performs the assigned portion of the matrix-vector multiplication.
- `init_mat_vec_mul`: Initializes the data structures and threads needed for the matrix-vector multiplication.
- `mat_vec_mul`: Performs the actual matrix-vector multiplication by distributing the workload among the threads.
- `close_mat_vec_mul`: Wraps up the matrix-vector multiplication by terminating the threads and freeing allocated resources.
- `main`: The entry point of the program. It loads the model, tokenizer, and performs the matrix-vector multiplication using the transformer model.

## Dependencies

The code relies on the following dependencies:

- POSIX threads (pthreads) library
- Standard C libraries: stdio.h, stdlib.h, math.h, sys/time.h, sys/resource.h
- Custom header file: utilities.h

## Performance

The parallel matrix-vector multiplication algorithm achieves improved performance by distributing the computation across multiple threads. The achieved tokens per second (tok/s) is reported at the end of the program execution.

## Development Platform

- Ubuntu 20.04 Docker Container


## Compilation Command

```
gcc -o llama2 llama2.c utilities.c -O2 -pthread -lm
```

Feel free to explore and utilize this parallel matrix-vector multiplication implementation in your projects!
