#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "CUFFT error in %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel for element-wise multiplication in frequency domain
__global__ void multiplySpectrum(cufftComplex* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float mag = cuCabsf(data[idx]);
        data[idx] = make_cuComplex(mag * mag, 0.0f);
    }
}

// Custom 3D FFT implementation
class FFT3D {
private:
    int nx, ny, nz;
    cufftHandle planFwd, planInv;
    cufftComplex *d_data;
    size_t dataSize;

public:
    FFT3D(int nx, int ny, int nz) : nx(nx), ny(ny), nz(nz) {
        dataSize = nx * ny * nz * sizeof(cufftComplex);

        CHECK_CUDA(cudaMalloc(&d_data, dataSize));

        CHECK_CUFFT(cufftPlan3d(&planFwd, nz, ny, nx, CUFFT_C2C));
        CHECK_CUFFT(cufftPlan3d(&planInv, nz, ny, nx, CUFFT_C2C));
    }

    ~FFT3D() {
        CHECK_CUDA(cudaFree(d_data));
        CHECK_CUFFT(cufftDestroy(planFwd));
        CHECK_CUFFT(cufftDestroy(planInv));
    }

    void forward(cufftComplex* h_data) {
        CHECK_CUDA(cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice));
        CHECK_CUFFT(cufftExecC2C(planFwd, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost));
    }

    void inverse(cufftComplex* h_data) {
        CHECK_CUDA(cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice));
        CHECK_CUFFT(cufftExecC2C(planInv, d_data, d_data, CUFFT_INVERSE));
        CHECK_CUDA(cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost));

        // Normalize
        float scale = 1.0f / (nx * ny * nz);
        for (int i = 0; i < nx * ny * nz; i++) {
            h_data[i].x *= scale;
            h_data[i].y *= scale;
        }
    }

    void powerSpectrum() {
        int blockSize = 256;
        int gridSize = (nx * ny * nz + blockSize - 1) / blockSize;
        multiplySpectrum<<<gridSize, blockSize>>>(d_data, nx * ny * nz);
    }
};

// Helper function to generate 3D sine wave
void generateSineWave(cufftComplex* data, int nx, int ny, int nz, float fx, float fy, float fz) {
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                float val = sinf(2 * M_PI * (fx * i / nx + fy * j / ny + fz * k / nz));
                data[k * ny * nx + j * nx + i] = make_cuComplex(val, 0.0f);
            }
        }
    }
}

int main() {
    const int nx = 128, ny = 128, nz = 128;
    cufftComplex *h_data = (cufftComplex*)malloc(nx * ny * nz * sizeof(cufftComplex));

    // Generate input data (3D sine wave)
    generateSineWave(h_data, nx, ny, nz, 5.0f, 3.0f, 2.0f);

    // Create FFT3D object
    FFT3D fft(nx, ny, nz);

    // Perform forward FFT
    fft.forward(h_data);

    // Compute power spectrum
    fft.powerSpectrum();

    // Perform inverse FFT
    fft.inverse(h_data);

    // Verify result (should be similar to input, but with some numerical error)
    float maxError = 0.0f;
    for (int i = 0; i < nx * ny * nz; i++) {
        float expected = sinf(2 * M_PI * (5.0f * (i % nx) / nx + 3.0f * ((i / nx) % ny) / ny + 2.0f * (i / (nx * ny)) / nz));
        float error = fabsf(h_data[i].x - expected);
        maxError = fmaxf(maxError, error);
    }

    printf("Max error: %f\n", maxError);

    free(h_data);
    return 0;
}