#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <getopt.h>

void printPlate (double* P, int width)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%.1f ", P[width * j + i]);
        }
        printf("\n");
    }
}

int main (int argc, char *argv[])
{
    int n, iter, opt;
    while ((opt = getopt(argc, argv, "n:I:")) != -1)
    {
        switch (opt)
        {
            case 'I':
                iter = atoi(optarg);
            case 'n':
                n = atoi(optarg);
                break;
            default:
                exit(EXIT_FAILURE);
        }
    }
    int width = n + 2;
    int size = width * width * sizeof(double);
    double *G, *H;
    cudaMallocManaged(&G, size);
    cudaMallocManaged(&H, size);
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (i == 0 || j == 0 || i == width - 1 || j == width - 1)
            {
                if (j == 0 && i > 0.3 * width && i < 0.7 * width)
                {
                    G[width * j + i] = 100.0;
                    H[width * j + i] = 100.0;
                }
                else
                {
                    G[width * j + i] = 20.0;
                    H[width * j + i] = 20.0;
                }
            }
            else
            {
                G[width * j + i] = 0.0;
                H[width * j + i] = 0.0;
            }
        }
    }
    printPlate(G, width);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop);
    int threads = sqrt(prop.maxThreadPerBlock);
    dim3 dimBlock(threads, threads);
    dim3 dimGrid(n/threads + 1, n/threads + 1);
    for (int i = 0; i < iter; i++)
    {
        cudaEvent_t stop;
        cudaEventCreate(&stop);
        kernel<<<dimGrid, dimBlock>>>(G, H, n);
        kernel<<<dimGrid, dimBlock>>>(H, G, n);
        cudaEventSynchronize(stop);
    }
    printPlate(G, width);
    return 0;
}

__global__ void kernel(double *G, double *H, int n, int width)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    printf("%f\n ", H[width * y + x]);
    if (x < n && y < n)
    {
        G[y * width + x] = 0.25 * (H[y * width + x + 1] + H[y * width + x - 1] + H[(y - 1) * width + x] + H[(y + 1) * width + x]);
    }
    printf("\n");
}