#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "bilateral_grid.h"
#include "bilateral_grid_gpu.h"
#ifndef NO_AUTO_SCHEDULE
#include "bilateral_grid_auto_schedule.h"
#endif

#include "halide_benchmark.h"
#include "HalideBuffer.h"
#include "halide_image_io.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: ./filter input.png output.png range_sigma timing_iterations\n"
               "e.g. ./filter input.png output.png 0.1 10\n");
        return 0;
    }

    float r_sigma = (float) atof(argv[3]);
    int timing_iterations = atoi(argv[4]);

    // Buffer<float> input = load_and_convert_image(argv[1]);
    printf("Warning: testing performance with input image ignored!\n");
    const int width = 6408;
    const int height = 4802;
    Buffer<float> input(width, height);
    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            input(x, y) =  static_cast<float>(rand()) / RAND_MAX;
        }
    }

    Buffer<float> output(input.width(), input.height());

    printf("Input image size: %dx%d\n", input.width(), input.height());

    bilateral_grid(input, r_sigma, output);

    // Timing code. Timing doesn't include copying the input data to
    // the gpu or copying the output back.

    // Manually-tuned version
    double min_t_manual = benchmark(timing_iterations, 10, [&]() {
        bilateral_grid(input, r_sigma, output);
    });
    printf("Manually-tuned time: %gms\n", min_t_manual * 1e3);

    #ifndef NO_AUTO_SCHEDULE
    // Auto-scheduled version
    double min_t_auto = benchmark(timing_iterations, 10, [&]() {
        bilateral_grid_auto_schedule(input, r_sigma, output);
    });
    printf("Auto-scheduled time: %gms\n", min_t_auto * 1e3);
    #endif

    bilateral_grid_gpu(input, r_sigma, output);

    double min_t_gpu = benchmark(timing_iterations, 10, [&]() {
        bilateral_grid_gpu(input, r_sigma, output);
    });
    printf("GPU-scheduled time: %gms\n", min_t_gpu * 1e3);
    output.copy_to_host();

    convert_and_save_image(output, argv[2]);

    return 0;
}
