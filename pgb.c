#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>  // For measuring computation time

#define KERNEL_SIZE 11
#define SIGMA 6.0

// Function to generate a Gaussian kernel
void generateGaussianKernel(float kernel[KERNEL_SIZE][KERNEL_SIZE], float sigma) {
    int halfSize = KERNEL_SIZE / 2;
    float sum = 0.0;
    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            kernel[i + halfSize][j + halfSize] = exp(-(i * i + j * j) / (2 * sigma * sigma));
            sum += kernel[i + halfSize][j + halfSize];
        }
    }

    // Normalize the kernel
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            kernel[i][j] /= sum;
        }
    }
}

// Function to apply the Gaussian blur
void gaussianBlur(unsigned char* input, unsigned char* output, int width, int height, int channels, float kernel[KERNEL_SIZE][KERNEL_SIZE]) {
    int halfSize = KERNEL_SIZE / 2;

    // Parallelize the loop with OpenMP
    #pragma omp parallel for collapse(2)
    for (int y = halfSize; y < height - halfSize; y++) {
        for (int x = halfSize; x < width - halfSize; x++) {
            for (int c = 0; c < channels; c++) { // Loop through color channels
                float sum = 0.0;

                // Apply the kernel
                for (int ky = -halfSize; ky <= halfSize; ky++) {
                    for (int kx = -halfSize; kx <= halfSize; kx++) {
                        int pixelIndex = ((y + ky) * width + (x + kx)) * channels + c;
                        sum += input[pixelIndex] * kernel[ky + halfSize][kx + halfSize];
                    }
                }

                // Write the blurred value to the output image
                output[(y * width + x) * channels + c] = (unsigned char)sum;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    // Load the image
    unsigned char* inputImage = stbi_load(argv[1], &width, &height, &channels, 0);
    if (inputImage == NULL) {
        printf("Error: Could not load image.\n");
        return 1;
    }

    // Allocate memory for the output image
    unsigned char* outputImage = (unsigned char*)malloc(width * height * channels);
    if (outputImage == NULL) {
        printf("Error: Could not allocate memory for the output image.\n");
        stbi_image_free(inputImage);
        return 1;
    }

    float kernel[KERNEL_SIZE][KERNEL_SIZE];
    generateGaussianKernel(kernel, SIGMA);

    // Start measuring the time
    clock_t start_time = clock();

    // Apply the Gaussian blur
    gaussianBlur(inputImage, outputImage, width, height, channels, kernel);

    // End measuring the time
    clock_t end_time = clock();
    
    // Calculate and print the time taken in seconds
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    // double time_spent1 = (double)(end_time - start_time);
    printf("Gaussian blur applied in %.6f seconds.\n", time_spent/10.0);

    // Save the blurred image
    if (!stbi_write_png(argv[2], width, height, channels, outputImage, width * channels)) {
        printf("Error: Could not save image.\n");
    } else {
        printf("Gaussian blur applied and image saved successfully.\n");
    }

    // Free memory
    stbi_image_free(inputImage);
    free(outputImage);

    return 0;
}
