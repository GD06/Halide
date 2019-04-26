#include "Halide.h"

#include "daubechies_constants.h"

namespace {

Halide::Var x("x"), y("y"), c("c"), xi("xi"), yi("yi");

class inverse_haar_x : public Halide::Generator<inverse_haar_x> {
public:
    Input<Buffer<float>> in_{"in" , 3};
    Output<Buffer<float>> out_{"out" , 2};

    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 8};

    void generate() {
        Func in = Halide::BoundaryConditions::repeat_edge(in_);

        out_(x, y) = select(x%2 == 0,
                           in(x/2, y, 0) + in(x/2, y, 1),
                           in(x/2, y, 0) - in(x/2, y, 1));
        out_.unroll(x, 2)
            .gpu_tile(x, y, xi, yi, tile_x, tile_y);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(inverse_haar_x, inverse_haar_x)
