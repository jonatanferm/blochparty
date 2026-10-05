// PyTorch binding for kernel.h: checks the tensors and hands their raw pointers over.
// The kernel itself runs without the GIL and without ATen.

#include <torch/extension.h>

#include "kernel.h"

namespace {

void check(const torch::Tensor& t, const char* name, torch::ScalarType dtype, std::vector<int64_t> shape) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be on the CPU");
    TORCH_CHECK(t.scalar_type() == dtype, name, " has dtype ", t.scalar_type(), ", expected ", dtype);
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.sizes() == torch::IntArrayRef(shape), name, " has shape ", t.sizes(), ", expected ",
                torch::IntArrayRef(shape));
}

// Returns the (n_adc, C, 2) signal; m (N, 3) is read as the start state and overwritten.
torch::Tensor simulate(torch::Tensor pos, torch::Tensor pd, torch::Tensor t1, torch::Tensor t2,
                       torch::Tensor db0, torch::Tensor sens, torch::Tensor dt, torch::Tensor rf,
                       torch::Tensor grad, torch::Tensor adc_phase, torch::Tensor flags, torch::Tensor m,
                       int64_t n_threads, int64_t block) {
    const auto dtype = pd.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kFloat64, "only float32 and float64 are supported");
    TORCH_CHECK(pd.dim() == 1 && sens.dim() == 3, "pd must be (N,) and sens (N, C, 2)");
    const int64_t n = pd.size(0), nc = sens.size(1), ns = dt.size(0);
    check(pos, "pos", dtype, {n, 3});
    check(pd, "pd", dtype, {n});
    check(t1, "t1", dtype, {n});
    check(t2, "t2", dtype, {n});
    check(db0, "db0", dtype, {n});
    check(sens, "sens", dtype, {n, nc, 2});
    check(dt, "dt", dtype, {ns});
    check(rf, "rf", dtype, {ns, 2});
    check(grad, "grad", dtype, {ns, 3});
    check(adc_phase, "adc_phase", dtype, {ns});
    check(flags, "flags", torch::kUInt8, {ns});
    check(m, "m", dtype, {n, 3});

    const auto* f = flags.data_ptr<uint8_t>();
    auto signal = torch::empty({blochparty::fused::count_adc(f, ns), nc, 2}, pd.options());
    AT_DISPATCH_FLOATING_TYPES(dtype, "blochparty_fused_simulate", [&] {
        using T = scalar_t;
        const blochparty::fused::Spins<T> spins{n, nc, pos.data_ptr<T>(), pd.data_ptr<T>(), t1.data_ptr<T>(),
                                                t2.data_ptr<T>(), db0.data_ptr<T>(), sens.data_ptr<T>()};
        const blochparty::fused::Segments<T> seq{ns, dt.data_ptr<T>(), rf.data_ptr<T>(), grad.data_ptr<T>(),
                                                 adc_phase.data_ptr<T>(), f};
        pybind11::gil_scoped_release no_gil;
        blochparty::fused::simulate<T>(spins, seq, m.data_ptr<T>(), signal.data_ptr<T>(),
                                       static_cast<int>(n_threads), static_cast<int>(block));
    });
    return signal;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("simulate", &simulate, "Fused forward Bloch simulation (see kernel.h)");
}
