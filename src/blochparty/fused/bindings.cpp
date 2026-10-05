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

struct Inputs {
    int64_t n, nc, ns;
    torch::ScalarType dtype;
};

Inputs check_inputs(const torch::Tensor& pos, const torch::Tensor& pd, const torch::Tensor& t1,
                    const torch::Tensor& t2, const torch::Tensor& db0, const torch::Tensor& sens,
                    const torch::Tensor& dt, const torch::Tensor& rf, const torch::Tensor& grad,
                    const torch::Tensor& adc_phase, const torch::Tensor& flags) {
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
    return {n, nc, ns, dtype};
}

template <typename T>
blochparty::fused::Spins<T> spins_of(const Inputs& in, torch::Tensor& pos, torch::Tensor& pd, torch::Tensor& t1,
                                      torch::Tensor& t2, torch::Tensor& db0, torch::Tensor& sens) {
    return {in.n, in.nc, pos.data_ptr<T>(), pd.data_ptr<T>(), t1.data_ptr<T>(),
            t2.data_ptr<T>(), db0.data_ptr<T>(), sens.data_ptr<T>()};
}

template <typename T>
blochparty::fused::Segments<T> segments_of(const Inputs& in, torch::Tensor& dt, torch::Tensor& rf,
                                            torch::Tensor& grad, torch::Tensor& adc_phase, torch::Tensor& flags) {
    return {in.ns, dt.data_ptr<T>(), rf.data_ptr<T>(), grad.data_ptr<T>(), adc_phase.data_ptr<T>(),
            flags.data_ptr<uint8_t>()};
}

// Returns the (n_adc, C, 2) signal and, if every > 0, the (n_ckpt, N, 3) checkpoints for
// adjoint() (else an empty tensor); m (N, 3) is read as the start state and overwritten.
std::vector<torch::Tensor> simulate(torch::Tensor pos, torch::Tensor pd, torch::Tensor t1, torch::Tensor t2,
                                    torch::Tensor db0, torch::Tensor sens, torch::Tensor dt, torch::Tensor rf,
                                    torch::Tensor grad, torch::Tensor adc_phase, torch::Tensor flags,
                                    torch::Tensor m, int64_t n_threads, int64_t block, int64_t every) {
    const Inputs in = check_inputs(pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, flags);
    check(m, "m", in.dtype, {in.n, 3});
    TORCH_CHECK(every >= 0, "checkpoint interval must be >= 0");
    const auto* f = flags.data_ptr<uint8_t>();
    auto signal = torch::empty({blochparty::fused::count_adc(f, in.ns), in.nc, 2}, pd.options());
    const int64_t n_ckpt = every ? blochparty::fused::count_checkpoints(in.ns, every) : 0;
    auto ckpt = torch::empty({n_ckpt, in.n, 3}, pd.options());
    AT_DISPATCH_FLOATING_TYPES(in.dtype, "blochparty_fused_simulate", [&] {
        using T = scalar_t;
        const auto spins = spins_of<T>(in, pos, pd, t1, t2, db0, sens);
        const auto seq = segments_of<T>(in, dt, rf, grad, adc_phase, flags);
        pybind11::gil_scoped_release no_gil;
        blochparty::fused::simulate<T>(spins, seq, m.data_ptr<T>(), signal.data_ptr<T>(),
                                       static_cast<int>(n_threads), static_cast<int>(block),
                                       every ? ckpt.data_ptr<T>() : nullptr, every);
    });
    return {signal, ckpt};
}

// Gradients through simulate(): returns (dt, rf, grad, adc_phase, pos, pd, t1, t2, db0, sens, m),
// each undefined (None) unless the matching entry of `needs` is set.
std::vector<torch::Tensor> adjoint(torch::Tensor pos, torch::Tensor pd, torch::Tensor t1, torch::Tensor t2,
                                   torch::Tensor db0, torch::Tensor sens, torch::Tensor dt, torch::Tensor rf,
                                   torch::Tensor grad, torch::Tensor adc_phase, torch::Tensor flags,
                                   torch::Tensor ckpt, int64_t every, torch::Tensor signal,
                                   std::optional<torch::Tensor> g_signal, std::optional<torch::Tensor> g_m,
                                   std::vector<bool> needs, int64_t n_threads, int64_t block) {
    const Inputs in = check_inputs(pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, flags);
    TORCH_CHECK(every >= 1, "checkpoint interval must be positive");
    const int64_t n_adc = blochparty::fused::count_adc(flags.data_ptr<uint8_t>(), in.ns);
    check(ckpt, "ckpt", in.dtype, {blochparty::fused::count_checkpoints(in.ns, every), in.n, 3});
    check(signal, "signal", in.dtype, {n_adc, in.nc, 2});
    if (g_signal) check(*g_signal, "g_signal", in.dtype, {n_adc, in.nc, 2});
    if (g_m) check(*g_m, "g_m", in.dtype, {in.n, 3});
    TORCH_CHECK(needs.size() == 11, "needs must have 11 entries");
    const std::vector<std::vector<int64_t>> shapes = {
        {in.ns}, {in.ns, 2}, {in.ns, 3}, {in.ns}, {in.n, 3}, {in.n}, {in.n}, {in.n}, {in.n}, {in.n, in.nc, 2}, {in.n, 3}};
    std::vector<torch::Tensor> out(11);
    for (size_t i = 0; i < out.size(); ++i)
        if (needs[i]) out[i] = torch::empty(shapes[i], pd.options());
    AT_DISPATCH_FLOATING_TYPES(in.dtype, "blochparty_fused_adjoint", [&] {
        using T = scalar_t;
        const auto spins = spins_of<T>(in, pos, pd, t1, t2, db0, sens);
        const auto seq = segments_of<T>(in, dt, rf, grad, adc_phase, flags);
        auto ptr = [&](size_t i) { return out[i].defined() ? out[i].data_ptr<T>() : nullptr; };
        blochparty::fused::Grads<T> g;
        g.dt = ptr(0), g.rf = ptr(1), g.grad = ptr(2), g.adc_phase = ptr(3), g.pos = ptr(4), g.pd = ptr(5);
        g.t1 = ptr(6), g.t2 = ptr(7), g.db0 = ptr(8), g.sens = ptr(9), g.m = ptr(10);
        const T* gs = g_signal ? g_signal->data_ptr<T>() : nullptr;
        const T* gm = g_m ? g_m->data_ptr<T>() : nullptr;
        pybind11::gil_scoped_release no_gil;
        blochparty::fused::adjoint<T>(spins, seq, ckpt.data_ptr<T>(), every, signal.data_ptr<T>(), gs, gm, g,
                                      static_cast<int>(n_threads), static_cast<int>(block));
    });
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("simulate", &simulate, "Fused forward Bloch simulation, optionally checkpointed (see kernel.h)");
    mod.def("adjoint", &adjoint, "Reverse sweep of the fused simulation (see kernel.h)");
}
