🚀 CUDA Batch Image

This project performs parallel Gaussian blurring on multiple images using CUDA, enabling faster processing by utilizing GPU acceleration.

🛠️ Compilation
make
▶️ Usage
./image_processor --input_dir <path_to_images> --output_dir <path_to_output>
📌 Example
./image_processor --input_dir ./data/images --output_dir ./data/output
📄 Project Overview

The application is designed to process a collection of images (tested on 10+ high-resolution 4K images) by applying a 3×3 Gaussian blur filter using a CUDA kernel. Input and output directories are provided via command-line arguments.

Key takeaways:

Proper tuning of CUDA grid and block dimensions significantly impacts performance
Handling image boundaries correctly inside the kernel is essential for accurate results
⚙️ Code Structure

The implementation is organized to efficiently leverage GPU parallelism for image processing tasks.

🔑 Core Components
1. Main Function
Reads command-line inputs (--input_dir, --output_dir)
Ensures the provided directories are valid
Invokes the processImages function to start processing
2. processImages Function
Iterates through .jpg and .png files using std::filesystem
For each image:
Loads image data via OpenCV (imread)
Allocates memory on the GPU for input and output buffers
Transfers image data from CPU to GPU
Executes the Gaussian blur kernel
Copies processed data back to CPU memory
Saves the output using OpenCV (imwrite)
3. gaussianBlurKernel (CUDA Kernel)
Implements a 2D parallel kernel, where each thread handles one pixel
Applies a fixed 3×3 Gaussian filter with normalized coefficients
Processes RGB channels independently
Prevents invalid memory access using boundary clamping (min/max checks)
Uses a 16×16 thread block configuration for execution
🔍 Implementation Highlights
⚡ Parallel Execution: Each pixel is processed independently by a CUDA thread
🧠 Dynamic Grid Sizing: Grid dimensions are adapted based on image resolution
📦 Memory Handling: Uses cudaMalloc and cudaMemcpy for efficient data transfer
⚠️ Error Handling: Validates image loading and directory paths
🧹 Code Quality: Structured and modular design following standard C++ practice
