
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
