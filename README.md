# GPU Accelerated Sobel Edge Detection using CUDA

## Overview

This project demonstrates GPU accelerated image processing using CUDA and NVIDIA GPUs. The application performs Sobel Edge Detection on grayscale images and compares the performance of CPU execution versus GPU execution.

The project highlights the advantages of parallel computing using CUDA kernels, GPU threads, and device memory management.

---

## Features

- CUDA-based Sobel Edge Detection
- CPU implementation for comparison
- GPU performance acceleration
- Parallel image processing
- CUDA kernel programming
- Execution time benchmarking
- Grayscale image processing

---

## Technologies Used

- CUDA
- C++
- NVIDIA GPU
- nvcc compiler
- Google Colab

---

## CUDA Concepts Demonstrated

- CUDA Kernels
- Thread Blocks
- Grid Configuration
- Device Memory Allocation
- Host to Device Memory Transfer
- Parallel Processing
- GPU Optimization

---

## Project Objective

The objective of this project is to demonstrate how GPU computing can accelerate image processing tasks using CUDA. Sobel Edge Detection is implemented on both CPU and GPU to compare execution speed and efficiency.

---

## Files Included

| File | Description |
|------|-------------|
| edge_detection.cu | Main CUDA source code |
| input.pgm | Input grayscale image |
| output_cpu.pgm | CPU processed image |
| output_gpu.pgm | GPU processed image |
| results.txt | Execution timing results |
| screenshots/ | Execution screenshots |
| presentation/ | Project presentation |

---

## How to Run in Google Colab

### Step 1: Enable GPU

Runtime → Change Runtime Type → GPU

---

### Step 2: Verify GPU

```bash
!nvidia-smi
```
### Step 3: Compile CUDA Program
```
!nvcc edge_detection.cu -o edge_detection
```

### Step 4: Run Program
!./edge_detection

### Program
```

---

# Full edge_detection.cu

```cpp
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define WIDTH 512
#define HEIGHT 512

unsigned char inputImage[WIDTH * HEIGHT];
unsigned char outputCPU[WIDTH * HEIGHT];
unsigned char outputGPU[WIDTH * HEIGHT];

__global__ void sobelKernel(unsigned char* input, unsigned char* output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < WIDTH - 1 && y > 0 && y < HEIGHT - 1)
    {
        int Gx =
            -input[(y - 1) * WIDTH + (x - 1)] -
            2 * input[y * WIDTH + (x - 1)] -
            input[(y + 1) * WIDTH + (x - 1)] +
            input[(y - 1) * WIDTH + (x + 1)] +
            2 * input[y * WIDTH + (x + 1)] +
            input[(y + 1) * WIDTH + (x + 1)];

        int Gy =
            -input[(y - 1) * WIDTH + (x - 1)] -
            2 * input[(y - 1) * WIDTH + x] -
            input[(y - 1) * WIDTH + (x + 1)] +
            input[(y + 1) * WIDTH + (x - 1)] +
            2 * input[(y + 1) * WIDTH + x] +
            input[(y + 1) * WIDTH + (x + 1)];

        int magnitude = sqrtf(Gx * Gx + Gy * Gy);

        if (magnitude > 255)
            magnitude = 255;

        output[y * WIDTH + x] = magnitude;
    }
}

void sobelCPU(unsigned char* input, unsigned char* output)
{
    for (int y = 1; y < HEIGHT - 1; y++)
    {
        for (int x = 1; x < WIDTH - 1; x++)
        {
            int Gx =
                -input[(y - 1) * WIDTH + (x - 1)] -
                2 * input[y * WIDTH + (x - 1)] -
                input[(y + 1) * WIDTH + (x - 1)] +
                input[(y - 1) * WIDTH + (x + 1)] +
                2 * input[y * WIDTH + (x + 1)] +
                input[(y + 1) * WIDTH + (x + 1)];

            int Gy =
                -input[(y - 1) * WIDTH + (x - 1)] -
                2 * input[(y - 1) * WIDTH + x] -
                input[(y - 1) * WIDTH + (x + 1)] +
                input[(y + 1) * WIDTH + (x - 1)] +
                2 * input[(y + 1) * WIDTH + x] +
                input[(y + 1) * WIDTH + (x + 1)];

            int magnitude = sqrt(Gx * Gx + Gy * Gy);

            if (magnitude > 255)
                magnitude = 255;

            output[y * WIDTH + x] = magnitude;
        }
    }
}

void readPGM(const char* filename, unsigned char* image)
{
    std::ifstream file(filename, std::ios::binary);

    std::string magic;
    file >> magic;

    int width, height, maxval;
    file >> width >> height >> maxval;
    file.ignore();

    file.read((char*)image, WIDTH * HEIGHT);

    file.close();
}

void writePGM(const char* filename, unsigned char* image)
{
    std::ofstream file(filename, std::ios::binary);

    file << "P5\n";
    file << WIDTH << " " << HEIGHT << "\n";
    file << "255\n";

    file.write((char*)image, WIDTH * HEIGHT);

    file.close();
}

int main()
{
    readPGM("input.pgm", inputImage);

    auto cpuStart = std::chrono::high_resolution_clock::now();

    sobelCPU(inputImage, outputCPU);

    auto cpuEnd = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> cpuTime = cpuEnd - cpuStart;

    unsigned char *d_input, *d_output;

    cudaMalloc((void**)&d_input, WIDTH * HEIGHT);
    cudaMalloc((void**)&d_output, WIDTH * HEIGHT);

    cudaMemcpy(d_input, inputImage, WIDTH * HEIGHT, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks(WIDTH / 16, HEIGHT / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    sobelKernel<<<blocks, threads>>>(d_input, d_output);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaMemcpy(outputGPU, d_output, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    writePGM("output_cpu.pgm", outputCPU);
    writePGM("output_gpu.pgm", outputGPU);

    std::ofstream result("results.txt");

    result << "CPU Time: " << cpuTime.count() << " ms\n";
    result << "GPU Time: " << gpuTime << " ms\n";

    result.close();

    std::cout << "CPU Time: " << cpuTime.count() << " ms\n";
    std::cout << "GPU Time: " << gpuTime << " ms\n";

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
```

### OUTPUT:

<img width="632" height="346" alt="Screenshot 2026-05-26 160905" src="https://github.com/user-attachments/assets/085a556b-3035-488c-8c83-c3749fc38966" />

<img width="348" height="307" alt="Screenshot 2026-05-26 160915" src="https://github.com/user-attachments/assets/e1772e69-1755-41d9-a2a8-58207895080c" />

<img width="291" height="352" alt="Screenshot 2026-05-26 160932" src="https://github.com/user-attachments/assets/5062f1ae-4f3f-427c-85ef-ec61705b3da3" />


### RESULT:
This project implements Sobel Edge Detection using CUDA GPU programming. The application compares CPU and GPU performance for image processing tasks and demonstrates how parallel computing with CUDA significantly accelerates edge detection operations. The project uses CUDA kernels, GPU threads, device memory management, and execution benchmarking to showcase real-world GPU computing applications.
