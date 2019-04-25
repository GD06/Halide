#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "linear_blur.h"
#include "linear_blur_gpu.h"
//#include "simple_blur.h"

#include "halide_benchmark.h"
#include "HalideBuffer.h"
#include "halide_image_io.h"

using namespace Halide::Tools;
using namespace Halide::Runtime;

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage: ./linear_blur use_linear input.png output.png\n");
        return 0;
    }

    int use_linear = atoi(argv[1]);

    Buffer<float> input_tmp = load_and_convert_image(argv[2]);
    Buffer<float> input(input_tmp.width() * 2, input_tmp.height() * 2, 3);

    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            for (int c = 0; c < 3; ++c) {
                input(x, y, c) = input_tmp(x % input_tmp.width(),
                        y % input_tmp.height(), c);
            }
        }
    }
    printf("Input image size: %d x %d\n", input.width(), input.height());

    Buffer<float> output = Buffer<float>::make_with_shape_of(input);

    if (!use_linear) {
        printf("WARNING: This modified version will run only for linear_blur!\n");
    }

    // Call either the simple or linear-corrected blur at runtime,
    // mainly to demonstrate how simple_blur can be used either standalone
    // or fused into another Generator.
    //if (use_linear) {
    //    printf("Using linear blur...\n");
    //    linear_blur(input, output);
    //} else {
    //    printf("Using simple blur...\n");
    //    simple_blur(input, input.width(), input.height(), output);
    //}

    const int timing_iterations = 20;

    linear_blur(input, output);

    double min_t_cpu = benchmark(timing_iterations, 1, [&]() {
        linear_blur(input, output);        
    });
    printf("CPU Auto-scheduled time: %gms\n", min_t_cpu * 1e3);

    linear_blur_gpu(input, output);

    double min_t_gpu = benchmark(timing_iterations, 1, [&]() {
        linear_blur_gpu(input, output);
    });
    output.copy_to_host();
    printf("GPU Manual-scheduled time: %gms\n", min_t_gpu * 1e3);

    convert_and_save_image(output, argv[3]);

    return 0;
}
