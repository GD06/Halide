#include "Halide.h"
//#include "linear_to_srgb.stub.h"
//#include "srgb_to_linear.stub.h"
//#include "simple_blur.stub.h"

namespace {

using namespace Halide;

struct LinearBlur : public Halide::Generator<LinearBlur> {
    Input<Buffer<float>>  input{"input", 3};
    Output<Buffer<float>> output{"output", 3};

    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 8};

    void generate() {
        Var x("x"), y("y"), c("c"), xi("xi"), yi("yi");

        Func repeated;
        repeated = BoundaryConditions::repeat_edge(input);

        Func linear;
        linear(x, y, c) = select(repeated(x, y, c) <= 0.04045f,
                                 repeated(x, y, c) / 12.92f,
                                 pow(((repeated(x, y, c) + .055f) / (1.0f + .055f)), 2.4f));

        Func blur_x;
        blur_x(x, y, c) = (linear(x, y, c) + linear(x+1, y, c) + linear(x+2, y, c))/3;

        Func blurred;
        blurred(x, y, c) = (blur_x(x, y, c) + blur_x(x, y+1, c) + blur_x(x, y+2, c))/3;
 
        Func srgb;
        srgb(x, y, c) = select(blurred(x, y, c) <= .0031308f,
                               blurred(x, y, c) * 12.92f,
                               (1 + .055f) * pow(blurred(x, y, c), 1.0f / 2.4f) - .055f);
 
        output(x, y, c) = srgb(x, y, c);

        if (auto_schedule) {
            input.dim(0).set_bounds_estimate(0, 1536)
                 .dim(1).set_bounds_estimate(0, 2560)
                 .dim(2).set_bounds_estimate(0, 4);
            output.estimate(x, 0, 1536)
                  .estimate(y, 0, 2560)
                  .estimate(c, 0, 4);
            // TODO(srj): set_bounds_estimate should work for Output<Buffer<>>, but does not
            // output.dim(0).set_bounds_estimate(0, 1536)
            //       .dim(1).set_bounds_estimate(0, 2560)
            //       .dim(2).set_bounds_estimate(0, 4);
        } else if (get_target().has_gpu_feature()) {
            int factor = sizeof(int) / sizeof(short);
            Var y_inner("y_inner");
            output.vectorize(x, factor)
                .split(y, y, y_inner, tile_y).reorder(y_inner, x).unroll(y_inner)
                .gpu_tile(x, y, xi, yi, tile_x, 1);
        } else {
            assert(false && "non-auto_schedule not supported.");
            abort();
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(LinearBlur, linear_blur)
