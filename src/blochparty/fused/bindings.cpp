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

void check_b1(const Inputs& in, const std::optional<torch::Tensor>& b1) {
    if (b1) check(*b1, "b1", in.dtype, {in.n});
}

// Field model: the uniform offset (S,) and the concomitant-field coefficients (N, 6 or 7).
void check_fields(const Inputs& in, const std::optional<torch::Tensor>& b0_offset,
                  const std::optional<torch::Tensor>& conc, int64_t conc_mode) {
    if (b0_offset) check(*b0_offset, "b0_offset", in.dtype, {in.ns});
    TORCH_CHECK(conc_mode >= 0 && conc_mode <= 2, "conc_mode must be 0, 1 or 2");
    TORCH_CHECK(!conc == !conc_mode, "conc is needed exactly when conc_mode is set");
    if (conc) check(*conc, "conc", in.dtype, {in.n, conc_mode == 2 ? 7 : 6});
}

// Magnetization transfer (Spins::bound): the pool (N, 5), the lineshape and its table, the
// RF frequency per segment; all or nothing.
struct Bound {
    std::optional<torch::Tensor> pool, log_f, rf_freq;
    int64_t kind = 0;
    double cutoff = 0, x0 = 0, h = 1, xmin = 0;
};

void check_bound(const Inputs& in, const Bound& b) {
    if (!b.pool) {
        TORCH_CHECK(!b.rf_freq, "rf_freq is only used with a bound pool");
        return;
    }
    check(*b.pool, "bound", in.dtype, {in.n, 5});
    TORCH_CHECK(b.rf_freq, "a bound pool needs rf_freq");
    check(*b.rf_freq, "rf_freq", in.dtype, {in.ns});
    TORCH_CHECK(b.kind == 1 || b.kind == 2, "lineshape must be 1 (Lorentzian) or 2 (super-Lorentzian)");
    if (b.kind == 2) {
        TORCH_CHECK(b.log_f, "the super-Lorentzian needs its table");
        TORCH_CHECK(b.log_f->dim() == 1 && b.log_f->size(0) >= 2, "the lineshape table must be (n >= 2,)");
        check(*b.log_f, "log_f", in.dtype, {b.log_f->size(0)});
    }
}

template <typename T>
void set_bound(blochparty::fused::Spins<T>& sp, blochparty::fused::Segments<T>& seq, Bound& b) {
    if (!b.pool) return;
    sp.bound = b.pool->data_ptr<T>();
    sp.line.kind = static_cast<int>(b.kind);
    sp.line.cutoff = T(b.cutoff), sp.line.x0 = T(b.x0), sp.line.h = T(b.h), sp.line.xmin = T(b.xmin);
    if (b.log_f) sp.line.log_f = b.log_f->data_ptr<T>(), sp.line.n = b.log_f->size(0);
    seq.rf_freq = b.rf_freq->data_ptr<T>();
}

template <typename T>
blochparty::fused::Spins<T> spins_of(const Inputs& in, torch::Tensor& pos, torch::Tensor& pd, torch::Tensor& t1,
                                      torch::Tensor& t2, torch::Tensor& db0, torch::Tensor& sens,
                                      std::optional<torch::Tensor>& b1, std::optional<torch::Tensor>& conc,
                                      int64_t conc_mode) {
    return {in.n, in.nc, pos.data_ptr<T>(), pd.data_ptr<T>(), t1.data_ptr<T>(),
            t2.data_ptr<T>(), db0.data_ptr<T>(), sens.data_ptr<T>(), b1 ? b1->data_ptr<T>() : nullptr,
            conc ? conc->data_ptr<T>() : nullptr, static_cast<int>(conc_mode)};
}

template <typename T>
blochparty::fused::Segments<T> segments_of(const Inputs& in, torch::Tensor& dt, torch::Tensor& rf,
                                            torch::Tensor& grad, torch::Tensor& adc_phase, torch::Tensor& flags,
                                            std::optional<torch::Tensor>& b0_offset) {
    return {in.ns, dt.data_ptr<T>(), rf.data_ptr<T>(), grad.data_ptr<T>(), adc_phase.data_ptr<T>(),
            flags.data_ptr<uint8_t>(), b0_offset ? b0_offset->data_ptr<T>() : nullptr};
}

// Returns the (n_adc, C, 2) signal and, if every > 0, the (n_ckpt, N, 3) checkpoints for
// adjoint() (else an empty tensor); m (N, 3) is read as the start state and overwritten.
// b1 (N,) is the optional transmit scale (None: 1); b0_offset (S,) the uniform offset per
// segment (None: 0); conc the concomitant-field coefficients for conc_mode 1 (N, 6) or 2 (N, 7), or None.
std::vector<torch::Tensor> simulate(torch::Tensor pos, torch::Tensor pd, torch::Tensor t1, torch::Tensor t2,
                                    torch::Tensor db0, torch::Tensor sens, torch::Tensor dt, torch::Tensor rf,
                                    torch::Tensor grad, torch::Tensor adc_phase, torch::Tensor flags,
                                    torch::Tensor m, int64_t n_threads, int64_t block, int64_t every,
                                    std::optional<torch::Tensor> b1, std::optional<torch::Tensor> b0_offset,
                                    std::optional<torch::Tensor> conc, int64_t conc_mode,
                                    std::optional<torch::Tensor> bound, std::optional<torch::Tensor> log_f,
                                    std::optional<torch::Tensor> rf_freq, int64_t lineshape, double cutoff,
                                    double x0, double h, double xmin) {
    const Inputs in = check_inputs(pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, flags);
    Bound bd{bound, log_f, rf_freq, lineshape, cutoff, x0, h, xmin};
    check_bound(in, bd);
    const int64_t nm = bound ? 4 : 3;
    check(m, "m", in.dtype, {in.n, nm});
    check_b1(in, b1);
    check_fields(in, b0_offset, conc, conc_mode);
    TORCH_CHECK(every >= 0, "checkpoint interval must be >= 0");
    const auto* f = flags.data_ptr<uint8_t>();
    auto signal = torch::empty({blochparty::fused::count_adc(f, in.ns), in.nc, 2}, pd.options());
    const int64_t n_ckpt = every ? blochparty::fused::count_checkpoints(in.ns, every) : 0;
    auto ckpt = torch::empty({n_ckpt, in.n, nm}, pd.options());
    AT_DISPATCH_FLOATING_TYPES(in.dtype, "blochparty_fused_simulate", [&] {
        using T = scalar_t;
        auto spins = spins_of<T>(in, pos, pd, t1, t2, db0, sens, b1, conc, conc_mode);
        auto seq = segments_of<T>(in, dt, rf, grad, adc_phase, flags, b0_offset);
        set_bound<T>(spins, seq, bd);
        pybind11::gil_scoped_release no_gil;
        blochparty::fused::simulate<T>(spins, seq, m.data_ptr<T>(), signal.data_ptr<T>(),
                                       static_cast<int>(n_threads), static_cast<int>(block),
                                       every ? ckpt.data_ptr<T>() : nullptr, every);
    });
    return {signal, ckpt};
}

// Gradients through simulate(): returns (dt, rf, grad, adc_phase, pos, pd, t1, t2, db0, sens, m, b1,
// b0_offset, conc, bound, rf_freq), each undefined (None) unless the matching entry of `needs` is set
// (b1, b0_offset, conc and the bound pool only if given).
std::vector<torch::Tensor> adjoint(torch::Tensor pos, torch::Tensor pd, torch::Tensor t1, torch::Tensor t2,
                                   torch::Tensor db0, torch::Tensor sens, torch::Tensor dt, torch::Tensor rf,
                                   torch::Tensor grad, torch::Tensor adc_phase, torch::Tensor flags,
                                   torch::Tensor ckpt, int64_t every, torch::Tensor signal,
                                   std::optional<torch::Tensor> g_signal, std::optional<torch::Tensor> g_m,
                                   std::vector<bool> needs, int64_t n_threads, int64_t block,
                                   std::optional<torch::Tensor> b1, std::optional<torch::Tensor> b0_offset,
                                   std::optional<torch::Tensor> conc, int64_t conc_mode,
                                   std::optional<torch::Tensor> bound, std::optional<torch::Tensor> log_f,
                                   std::optional<torch::Tensor> rf_freq, int64_t lineshape, double cutoff,
                                   double x0, double h, double xmin) {
    const Inputs in = check_inputs(pos, pd, t1, t2, db0, sens, dt, rf, grad, adc_phase, flags);
    check_b1(in, b1);
    check_fields(in, b0_offset, conc, conc_mode);
    Bound bd{bound, log_f, rf_freq, lineshape, cutoff, x0, h, xmin};
    check_bound(in, bd);
    const int64_t nm = bound ? 4 : 3;
    TORCH_CHECK(every >= 1, "checkpoint interval must be positive");
    const int64_t n_adc = blochparty::fused::count_adc(flags.data_ptr<uint8_t>(), in.ns);
    check(ckpt, "ckpt", in.dtype, {blochparty::fused::count_checkpoints(in.ns, every), in.n, nm});
    check(signal, "signal", in.dtype, {n_adc, in.nc, 2});
    if (g_signal) check(*g_signal, "g_signal", in.dtype, {n_adc, in.nc, 2});
    if (g_m) check(*g_m, "g_m", in.dtype, {in.n, nm});
    TORCH_CHECK(needs.size() == 16, "needs must have 16 entries");
    TORCH_CHECK(b1 || !needs[11], "the b1 gradient needs b1");
    TORCH_CHECK(b0_offset || !needs[12], "the b0_offset gradient needs b0_offset");
    TORCH_CHECK(conc || !needs[13], "the conc gradient needs conc");
    TORCH_CHECK(bound || (!needs[14] && !needs[15]), "the bound-pool gradients need a bound pool");
    const std::vector<std::vector<int64_t>> shapes = {{in.ns}, {in.ns, 2}, {in.ns, 3}, {in.ns},
                                                      {in.n, 3}, {in.n}, {in.n}, {in.n}, {in.n}, {in.n, in.nc, 2},
                                                      {in.n, nm}, {in.n}, {in.ns}, {in.n, conc_mode == 2 ? 7 : 6},
                                                      {in.n, 5}, {in.ns}};
    std::vector<torch::Tensor> out(16);
    for (size_t i = 0; i < out.size(); ++i)
        if (needs[i]) out[i] = torch::empty(shapes[i], pd.options());
    AT_DISPATCH_FLOATING_TYPES(in.dtype, "blochparty_fused_adjoint", [&] {
        using T = scalar_t;
        auto spins = spins_of<T>(in, pos, pd, t1, t2, db0, sens, b1, conc, conc_mode);
        auto seq = segments_of<T>(in, dt, rf, grad, adc_phase, flags, b0_offset);
        set_bound<T>(spins, seq, bd);
        auto ptr = [&](size_t i) { return out[i].defined() ? out[i].data_ptr<T>() : nullptr; };
        blochparty::fused::Grads<T> g;
        g.dt = ptr(0), g.rf = ptr(1), g.grad = ptr(2), g.adc_phase = ptr(3), g.pos = ptr(4), g.pd = ptr(5);
        g.t1 = ptr(6), g.t2 = ptr(7), g.db0 = ptr(8), g.sens = ptr(9), g.m = ptr(10), g.b1 = ptr(11);
        g.b0_offset = ptr(12), g.conc = ptr(13), g.bound = ptr(14), g.rf_freq = ptr(15);
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
