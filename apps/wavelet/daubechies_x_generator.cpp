#include "Halide.h"

#include "daubechies_constants.h"

namespace {

Halide::Var x("x"), y("y"), c("c"), xi("xi"), yi("yi");

class daubechies_x : public Halide::Generator<daubechies_x> {
public:
    Input<Buffer<float>> in_{"in" , 2};
    Output<Buffer<float>> out_{"out" , 3};

    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 8};

    void generate() {
        Func in = Halide::BoundaryConditions::repeat_edge(in_);

        out_(x, y, c) = select(c == 0,
                              D0*in(2*x-1, y) + D1*in(2*x, y) + D2*in(2*x+1, y) + D3*in(2*x+2, y),
                              D3*in(2*x-1, y) - D2*in(2*x, y) + D1*in(2*x+1, y) - D0*in(2*x+2, y));
        out_.unroll(c, 2)
            .gpu_tile(x, y, xi, yi, tile_x, tile_y);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(daubechies_x, daubechies_x)
