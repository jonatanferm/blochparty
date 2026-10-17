"""Anatomy assembly vs conservation, Ernst, continuous-box FT and sphere fields.

All unit fixtures are synthetic; real BrainWeb checks and throughput live in
validation/anatomy/anatomy_brainweb.py and validation/anatomy/RESULTS.md.
"""
import gzip
import hashlib
import json
import math
from dataclasses import fields, replace

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty.anatomy import (
    AIR_CHI,
    BRAINWEB_TISSUES,
    DEFAULT_TISSUES,
    Anatomy,
    Distribution,
    anatomy_field_maps,
    load_brainweb,
    to_phantom,
)
from blochparty.maps import sample
from blochparty.phantom import b0_sphere

from cases import sphere


def mixed(shape=(3,4,5),dtype=torch.float64):
    wm = torch.linspace(.1,.8,math.prod(shape),dtype=dtype).reshape(shape)
    return Anatomy(dict(wm=wm,gm=.9-wm),(.001,.002,.003),(-.008,-.005,-.002))


@pytest.mark.parametrize("counts", [1,8,(1,2,3)])
def test_per_tissue_pd_conservation(counts):
    a = mixed()
    ph = to_phantom(a,spins_per_voxel=counts,chi_to_b0=False,t2star="off")
    for name,fraction in a.fractions.items():
        p = DEFAULT_TISSUES[3.][name]
        selected = torch.isclose(ph.t1, ph.t1.new_tensor(p.t1.mean), atol=1e-14, rtol=0)
        assert abs(ph.pd[selected].sum()-fraction.sum()*p.pd.mean) < 1e-12
        indices = torch.floor((ph.pos[selected]-torch.tensor(a.origin))/torch.tensor(a.voxel).flip(0)+.5).long()
        linear = (indices[:,2]*a.shape[1]+indices[:,1])*a.shape[2]+indices[:,0]
        pd_voxels = torch.zeros(math.prod(a.shape),dtype=ph.pd.dtype).scatter_add_(0,linear,ph.pd[selected])
        torch.testing.assert_close(pd_voxels.reshape(a.shape),fraction*p.pd.mean,atol=1e-12,rtol=0)


@pytest.mark.parametrize("voxel", [(.002,.004,.006),(.0008,.0013,.0021),(.0017,.0031,.0047)])
@pytest.mark.parametrize("dtype", [torch.float32,torch.float64])
def test_resample_conserves_volumes(voxel,dtype):
    a = mixed(dtype=dtype)
    b = a.resample(voxel)
    for name in a.fractions:
        before = a.fractions[name].double().sum()*math.prod(a.voxel)
        after = b.fractions[name].double().sum()*math.prod(b.voxel)
        assert abs(after/before-1) < 1e-6
    for i in range(3):
        assert b.origin[i]-voxel[2-i]/2 == pytest.approx(a.origin[i]-a.voxel[2-i]/2)


def test_cell_overlap_exact_and_crop_slab():
    v = torch.tensor([[[1.,0.,0.]]],dtype=torch.float64)
    a = Anatomy({"wm":v},(1,1,1),(0,0,0),chi=v*1e-6,maps={"M0":v})
    b = a.resample((1,1,.6))
    torch.testing.assert_close(b.fractions["wm"],torch.tensor([[[1.,2/3,0,0,0]]],dtype=v.dtype))
    c = mixed().crop((slice(1,3),slice(1,3),slice(2,5)))
    assert c.shape == (2,2,3)
    assert c.origin == pytest.approx((-.002,-.003,-.001))
    assert mixed().slab(-.0015,.001).shape[0] == 1
    with pytest.raises(ValueError):
        mixed().slab(1,.1)


def test_ernst_signal_of_mixed_voxels():
    a = mixed((1,1,5))
    ph = to_phantom(a,spins_per_voxel=1,jitter=None,chi_to_b0=False,t2star="off")
    tr,te,alpha = .02,.004,math.radians(25)
    e = torch.exp(-tr/ph.t1)
    initial = torch.zeros_like(ph.pos)
    initial[:,2] = ph.pd*(1-e)/(1-e*math.cos(alpha))
    x = torch.round((ph.pos[:,0]-a.origin[0])/a.voxel[2]).long()
    coils = torch.nn.functional.one_hot(x,5).to(torch.complex128)
    q = bp.SequenceBuilder().rf(1e-9,flip_deg=25,n_steps=1).delay(te-1e-9).readout(1,1e-9).build()
    actual = bp.simulate(ph,q,coils,m_init=initial).signal[:,0]
    expected = torch.zeros(5,dtype=torch.float64)
    for name,fraction in a.fractions.items():
        p = DEFAULT_TISSUES[3][name]
        e1 = math.exp(-tr/p.t1.mean)
        expected += fraction.flatten()*p.pd.mean*(1-e1)*math.sin(alpha)/(1-e1*math.cos(alpha))*math.exp(-te/p.t2.mean)
    assert (actual-1j*expected).abs().max() < 1e-8


def test_continuous_box_kspace_converges():
    # Anatomy boundary is shifted by 0.3 mm relative to a nominal reconstruction
    # grid. A membership cannot encode an unknown sub-cell boundary: resolve it
    # on this independent anatomy grid instead.
    a = Anatomy({"wm":torch.ones(1,1,7,dtype=torch.float64)},(.001,)*3,(.0003,0,0))
    frequencies = torch.linspace(0,800,61,dtype=torch.float64)
    p = DEFAULT_TISSUES[3]["wm"]
    expected = 7*p.pd.mean*torch.sinc(.007*frequencies)*torch.exp(-2j*math.pi*.0033*frequencies)
    errors = []
    for k in (1,4,16,64):
        ph = to_phantom(a,spins_per_voxel=(1,1,k),jitter=None,chi_to_b0=False,t2star="off")
        signal = torch.exp(-2j*math.pi*frequencies[:,None]*ph.pos[:,0]) @ ph.pd.to(torch.complex128)
        errors.append(((signal-expected).abs().max()/(7*p.pd.mean)).item())
    print("continuous box max normalized k-space error:",errors)
    assert all(b < a/8 for a,b in zip(errors[1:],errors[2:]))
    assert errors[-1] < 3e-5 and errors[0] > .05


def test_seed_controls_jitter_without_changing_properties():
    a = mixed()
    options = dict(chi_to_b0=False,t2star="table",spins_per_voxel=8)
    p,q,r = [to_phantom(a,seed=seed,**options) for seed in (12,12,13)]
    for name in ("pos","pd","t1","t2","db0"):
        assert torch.equal(getattr(p,name),getattr(q,name))
    assert not torch.equal(p.pos,r.pos)
    for name in ("pd","t1","t2","db0"):
        assert torch.equal(getattr(p,name),getattr(r,name))


def test_categorical_is_unbiased_including_air():
    a = Anatomy({"wm":torch.full((1,1,1),.3,dtype=torch.float64),
                 "gm":torch.full((1,1,1),.4,dtype=torch.float64)},(.001,)*3,(0,0,0))
    ph = to_phantom(a,sampling="categorical",spins_per_voxel=(1,1,50000),
                    chi_to_b0=False,t2star="off",seed=41)
    for name,fraction in a.fractions.items():
        p = DEFAULT_TISSUES[3][name]
        estimated = ph.pd[torch.isclose(ph.t1, ph.t1.new_tensor(p.t1.mean), atol=1e-14, rtol=0)].sum()/p.pd.mean
        assert abs(estimated-fraction.item()) < .008
    assert len(ph) < 50000


def test_b0_sphere_at_spin_positions_and_scanner_split():
    chi,pos,h,radius,_ = sphere(40)
    r = pos.norm(dim=-1)
    tissue = ((r < .7*radius) | ((r > 1.5*radius) & (r < 2.2*radius))).double()
    a = Anatomy({"wm":tissue},(h,)*3,tuple(pos[0,0,0].tolist()),chi=chi+AIR_CHI)
    pair = anatomy_field_maps(a,shim_order=None,padding=3)
    ph = to_phantom(a,spins_per_voxel=1,field_maps=pair,shim_order=None,t2star="off")
    expected = b0_sphere(ph.pos,(0,0,0),radius,9.4e-6)
    error = (ph.db0-expected).square().mean().sqrt().item()
    print("anatomical sphere RMS Hz:",error)
    assert error < 1.
    pair = anatomy_field_maps(a,shim_order=2)
    ph = to_phantom(a,spins_per_voxel=1,field_maps=pair,t2star="off")
    unshimmed = sample(pair[0].b0[None],ph.pos.new_tensor(a.origin),ph.pos.new_tensor(a.voxel),ph.pos)[:,0]
    correction = sample(pair[1].b0[None],ph.pos.new_tensor(a.origin),ph.pos.new_tensor(a.voxel),ph.pos)[:,0]
    torch.testing.assert_close(ph.db0,unshimmed+correction)
    # Crop spins only: full-head B0 and scanner map are retained.
    cropped = to_phantom(a,spins_per_voxel=1,field_maps=pair,t2star="off",
                         fov_region=(slice(5,12),slice(5,20),slice(5,20)))
    assert (cropped.pos[:,2] < a.origin[2]+11.5*h).all()


def test_t2prime_quantile_fid_and_no_double_count():
    a = Anatomy({"wm":torch.ones(1,1,1,dtype=torch.float64)},(.001,)*3,(0,0,0))
    ph = to_phantom(a,spins_per_voxel=(1,1,4096),chi_to_b0=False,t2star="table")
    t = torch.linspace(.001,.05,50,dtype=torch.float64)
    signal = torch.exp(-t[:,None]/ph.t2-2j*math.pi*t[:,None]*ph.db0) @ ph.pd.to(torch.complex128)/ph.pd.sum()
    reference = torch.exp(-t/DEFAULT_TISSUES[3]["wm"].t2star.mean)
    assert (signal-reference).abs().max() < .005
    with pytest.raises(ValueError,match="calibration"):
        to_phantom(a,chi_to_b0=True,shim_order=None,t2star="table")
    pair = (bp.FieldMaps((0,0,0),(.001,)*3,b0=torch.zeros(1,1,1,dtype=torch.float64)),)*2
    assert not to_phantom(a,field_maps=pair,t2star="off").db0.any()
    # a flat field has no macroscopic part: 'auto' (= 'separate') puts the whole R2* - R2 on the spins
    sep = to_phantom(a,field_maps=pair,spins_per_voxel=(1,1,64))
    full = to_phantom(a,chi_to_b0=False,spins_per_voxel=(1,1,64),t2star="table")
    assert torch.allclose(sep.db0,full.db0)
    a.maps["intrinsic_r2prime"] = torch.full((1,1,1),8.,dtype=torch.float64)
    micro = to_phantom(a,field_maps=pair)
    assert micro.db0.abs().max() > 0


def test_mt_diffusion_and_fat_bookkeeping():
    a = mixed()
    ph = to_phantom(a,mt=True,diffusion=True,chi_to_b0=False,t2star="off")
    for name in a.fractions:
        p = DEFAULT_TISSUES[3][name]
        selected = torch.isclose(ph.t1, ph.t1.new_tensor(p.t1.mean), atol=1e-14, rtol=0)
        assert torch.allclose(ph.bound.f[selected],torch.full_like(ph.pd[selected],p.bound_fraction.mean/(1-p.bound_fraction.mean)))
        assert (ph.diffusion[selected] == p.diffusivity.mean).all()
    fat = Anatomy({"fat":torch.ones(1,1,1,dtype=torch.float64)},(.001,)*3,(0,0,0))
    ph = to_phantom(fat,spins_per_voxel=1,chi_to_b0=False,t2star="off")
    assert len(ph) == 6
    assert ph.pd.sum() == pytest.approx(1.)
    assert (ph.db0 < 0).any() and (ph.db0 > 0).any()


CLASSES = ("background", "csf", "gm", "wm", "fat", "muscle", "skin", "skull", "vessels", "connective", "dura", "marrow")


def all_classes(n=2):
    return Anatomy({k: torch.full((n, n, n), 1 / 12, dtype=torch.float64) for k in CLASSES}, (.001,) * 3, (0, 0, 0))


def test_every_signalling_class_has_mt_and_diffusion():
    for strength in (1.5, 3.):
        for name, p in DEFAULT_TISSUES[strength].items():
            if p.pd.mean == 0:
                continue
            assert p.bound_fraction is not None and p.diffusivity is not None, name
            if p.bound_fraction.mean > 0:
                assert p.exchange_rate is not None and p.bound_t2 is not None, name
                # MT measured at 3 T is an assumption elsewhere
                assert p.bound_fraction.assumed or strength == 3., name
        ph = to_phantom(all_classes(), field_T=strength, mt=True, diffusion=True, chi_to_b0=False, t2star="off")
        assert ph.bound is not None and (ph.diffusion > 0).all()
    wm = DEFAULT_TISSUES[3.]["wm"]
    assert not wm.bound_fraction.assumed and not wm.exchange_rate.assumed and not wm.diffusivity.assumed
    assert DEFAULT_TISSUES[7.]["wm"].bound_fraction.assumed
    # CSF: Holz et al.'s Speedy-Angell fit; 2.299e-9 m^2/s at 25 C is their tabulated value
    holz = lambda celsius: 1.635e-8 * ((celsius + 273.15) / 215.05 - 1) ** 2.063
    assert holz(25) == pytest.approx(2.299e-9, rel=2e-4)
    assert DEFAULT_TISSUES[3.]["csf"].diffusivity.mean == pytest.approx(holz(37), rel=1e-3)


def test_loader_real_format_on_tiny_files(tmp_path):
    names = DEFAULT_TISSUES[3].keys()
    manifest = dict(schema=1,subject=4,shape=[3,4,5],voxel=[.0005]*3,
                    origin=[-.09025,-.12625,-.07225],dtype="<u2",scale=4095,files={})
    for name in names:
        values = np.full((3,4,5),4095 if name == "wm" else 0,dtype="<u2")
        if name == "wm":
            values.flat[0] = 1024  # deficit must remain air, not become pure WM
        if name == "gm":
            values.flat[1] = 4095  # overfull WM+GM must normalize jointly
        raw = values.tobytes()
        path = tmp_path/f"{name}.raws.gz"
        path.write_bytes(gzip.compress(raw))
        manifest["files"][name] = dict(path=path.name,sha256=hashlib.sha256(raw).hexdigest())
    (tmp_path/"subject04.json").write_text(json.dumps(manifest))
    native = load_brainweb(4,tmp_path,dtype=torch.float64)
    assert native.fractions["wm"].flatten()[0] == pytest.approx(1024/4095)
    assert native.fractions["wm"].flatten()[1] == .5
    assert native.fractions["gm"].flatten()[1] == .5
    a = load_brainweb(4,tmp_path,voxel=(.001,)*3,dtype=torch.float64)
    assert a.shape == (2,2,3)
    assert a.origin == pytest.approx((-.09,-.126,-.072))
    assert a.fractions["wm"].sum()*8 == pytest.approx(58.5+1024/4095)
    assert a.fractions["gm"].sum()*8 == pytest.approx(.5)
    (tmp_path/"wm.raws.gz").write_bytes(gzip.compress(bytes(120)))
    with pytest.raises(ValueError,match="checksum"):
        load_brainweb(4,tmp_path)


def test_tables_have_provenance_and_no_silent_7t_fallback():
    for strength,table in DEFAULT_TISSUES.items():
        for name,p in table.items():
            for f in fields(p):
                d = getattr(p,f.name)
                if d is not None:
                    assert isinstance(d,Distribution) and d.citation
    assert DEFAULT_TISSUES[3]["wm"].t1.sd == .045
    assert DEFAULT_TISSUES[7]["wm"].t1.mean == 1.19
    assert not DEFAULT_TISSUES[3]["csf"].t1.assumed and DEFAULT_TISSUES[3]["csf"].t2.mean == 2.03
    assert DEFAULT_TISSUES[1.5]["csf"].t2.assumed  # the 3 T value carried to 1.5 T
    assert DEFAULT_TISSUES[3]["fat"].t1.mean == .382 and DEFAULT_TISSUES[3]["fat"].t2.sd == .004
    assert not BRAINWEB_TISSUES[1.5]["csf"].t1.assumed
    a = Anatomy({"csf":torch.ones(1,1,1,dtype=torch.float64)},(.001,)*3,(0,0,0))
    with pytest.raises(ValueError,match="missing tissue"):
        to_phantom(a,field_T=7,chi_to_b0=False)


def test_invalid_anatomy_and_options():
    a = mixed()
    with pytest.raises(ValueError):
        Anatomy({"wm":torch.full((2,3,4),1.1)},(.001,)*3,(0,0,0))
    with pytest.raises(ValueError):
        Anatomy(a.fractions,(0,.001,.001),(0,0,0))
    with pytest.raises(ValueError):
        a.crop((slice(None,None,2),slice(None),slice(None)))
    assert torch.equal(to_phantom(a,variability={},chi_to_b0=False).t1,to_phantom(a,chi_to_b0=False).t1)
    with pytest.raises(ValueError,match="nonnegative"):
        to_phantom(a,variability={"t1":-.1,"length":.01},chi_to_b0=False)
    with pytest.raises(ValueError):
        to_phantom(a,spins_per_voxel=2)


def test_gpu_equals_core(gpu):
    a = mixed((3,3,3))
    ph = to_phantom(a,mt=True,diffusion=True,spins_per_voxel=8)
    q = bp.SequenceBuilder().rf(.001,flip_deg=35,n_steps=50).readout(12,.0002).build()
    ref = bp.simulate(ph,q).signal
    actual = gpu.simulate(ph,q).signal
    error = (actual-ref).abs().max()/ref.abs().max()
    print("anatomy GPU relative signal error:",error.item())
    assert error < 1e-5


@pytest.mark.slow
def test_variability_statistics_and_defaults():
    # a 96^3 one-tissue anatomy at 1 mm, L = 4 mm: the realised log-SD, mean and correlation at
    # lag L match the request within 10 %; T1 / T2 / PD fields are independent; default is off
    a = Anatomy({"wm": torch.ones(96, 96, 96, dtype=torch.float64)}, (.001,)*3, (0, 0, 0))
    var = {"t1": .1, "t2": .05, "pd": .02, "length": .004, "seed": 3}
    base = to_phantom(a, chi_to_b0=False, t2star="off", spins_per_voxel=1, jitter=None)
    ph = to_phantom(a, chi_to_b0=False, t2star="off", spins_per_voxel=1, jitter=None, variability=var)
    assert torch.equal(base.pos, ph.pos)
    for key, sd in (("t1", .1), ("t2", .05), ("pd", .02)):
        f = (getattr(ph, key) / getattr(base, key)).reshape(96, 96, 96)
        assert abs(float(f.mean()) - 1) < .01
        assert abs(float(f.std()) / sd - 1) < .1, (key, float(f.std()))
        g = f.log() - f.log().mean()
        rho = float((g[:, :, :-4] * g[:, :, 4:]).mean() / g.var())
        assert abs(rho / math.exp(-.5) - 1) < .1, (key, rho)
    c = torch.corrcoef(torch.stack([(ph.t1 / base.t1).log(), (ph.pd / base.pd).log()]))[0, 1]
    assert abs(float(c)) < .05
    # same variability seed, other jitter seed: same texture; other variability seed: different
    again = to_phantom(a, chi_to_b0=False, t2star="off", spins_per_voxel=1, jitter=None, seed=9, variability=var)
    assert torch.equal(again.t1, ph.t1)
    other = to_phantom(a, chi_to_b0=False, t2star="off", spins_per_voxel=1, jitter=None, variability=var | {"seed": 4})
    assert not torch.equal(other.t1, ph.t1)
    # crops see the same realisation as the full anatomy
    cut = to_phantom(a, chi_to_b0=False, t2star="off", spins_per_voxel=1, jitter=None, variability=var,
                     fov_region=(slice(10, 12), slice(None), slice(None)))
    assert torch.allclose(cut.t1, ph.t1.reshape(96, 96, 96)[10:12].reshape(-1))
    with pytest.raises(ValueError, match="variability"):
        to_phantom(a, chi_to_b0=False, variability={"t1": .1})


def test_float64_resampling_keeps_fractions_valid():
    # cumulative-sum resampling of complementary classes along long axes rounds by ~n eps per
    # class; in float64 that exceeded the 16 eps sum check (BrainWeb at 1.5 mm, float64, failed)
    g = torch.Generator().manual_seed(0)
    a_ = torch.rand(30, 400, 37, generator=g, dtype=torch.float64)
    b_ = torch.rand(30, 400, 37, generator=g, dtype=torch.float64) * (1 - a_)
    a = Anatomy({"wm": a_, "gm": b_, "background": 1 - a_ - b_}, (.0005,)*3, (0, 0, 0))
    for voxel in ((.0015,)*3, (.0007, .0013, .0011)):
        r = a.resample(voxel)
        assert float(sum(r.fractions.values()).max()) <= 1 + 16 * torch.finfo(torch.float64).eps
        for k in a.fractions:
            assert float(r.fractions[k].sum()) * math.prod(voxel) == pytest.approx(
                float(a.fractions[k].sum()) * .0005**3, rel=1e-6)


@pytest.mark.parametrize("grad_hz_per_mm", [3.0, 5.0, 8.0])
def test_measured_r2star_round_trip_without_double_counting(grad_hz_per_mm):
    # A tissue whose table T2* came from a 1 mm (in plane), 3-66 ms multi-echo fit, in a voxel-scale field
    # gradient: the assembled phantom (microscopic R2' on the spins + the field at their positions)
    # must give back that measured R2* when "acquired" the same way. Without the separation the
    # field is counted twice ('table' with no macroscopic term); dropping R2' ('off') undercounts.
    from blochparty.anatomy import DEFAULT_R2STAR_ACQUISITION, estimate_macroscopic_r2prime
    g = grad_hz_per_mm * 1e3  # Hz/m along x
    a = Anatomy({"wm": torch.ones(3, 3, 3, dtype=torch.float64)}, (.001,)*3, (-.001, -.001, -.001))
    x = torch.arange(3, dtype=torch.float64)[None, None, :].expand(3, 3, 3) * .001 - .001
    pair = (bp.FieldMaps(a.origin, (.001,)*3, b0=g * x), bp.FieldMaps(a.origin, (.001,)*3, b0=torch.zeros(3, 3, 3, dtype=torch.float64)))
    te = torch.tensor(DEFAULT_R2STAR_ACQUISITION["te"], dtype=torch.float64)
    est = estimate_macroscopic_r2prime(pair[0].b0, a.voxel)
    r2s_in = 1 / DEFAULT_TISSUES[3]["wm"].t2star.mean

    def measured(t2star, **kw):
        ph = to_phantom(a, field_maps=pair, spins_per_voxel=(1, 1, 8192), t2star=t2star,
                        fov_region=(slice(1, 2),)*3, **kw)
        sig = torch.exp(-te[:, None] / ph.t2 - 2j * math.pi * te[:, None] * ph.db0) @ ph.pd.to(torch.complex128)
        y = sig.abs().log()
        tc = te - te.mean()
        return float(-(tc * (y - y.mean())).sum() / tc.square().sum())

    sinc_rate = float(-(torch.sinc(g * .001 * te).abs().log() * (te - te.mean())).sum() / (te - te.mean()).square().sum())
    assert float(est[1, 1, 1]) == pytest.approx(sinc_rate, rel=1e-9)  # the estimator is the closed form
    out = measured("auto")
    r2 = 1 / DEFAULT_TISSUES[3]["wm"].t2.mean
    if sinc_rate < r2s_in - r2:  # consistent: 1.0 and 2.9 /s of the 4.9 /s R2* - R2
        assert out == pytest.approx(r2s_in, rel=0.02), (out, r2s_in)
        assert measured("table", macroscopic_r2prime=0.0) > r2s_in + 0.5 * sinc_rate  # double counted
        assert measured("off") < r2s_in - 1.0  # microscopic R2' lost
    else:  # 8 Hz/mm alone dephases faster (7.7 /s) than R2* - R2 allows: R2' clips at 0
        assert out == pytest.approx(measured("off"), rel=1e-12) and out > r2s_in


def test_systematic_sampling_counts_and_unbiased_pd():
    # docs/PERFORMANCE.md 2.6: floor or ceil of K f spins per tissue and voxel, PD unbiased over seeds
    g = torch.Generator().manual_seed(1)
    f = torch.rand(3, 4, 4, 4, generator=g, dtype=torch.float64)
    f = f / f.sum(0) * torch.rand(1, 4, 4, 4, generator=g, dtype=torch.float64).clamp(0.6, 1.0)  # some air
    a = Anatomy(dict(wm=f[0], gm=f[1], csf=f[2]), (.001,) * 3, (0., 0., 0.))
    k = 8
    totals = []
    for seed in range(200):
        ph = to_phantom(a, spins_per_voxel=(2, 2, 2), sampling="systematic", seed=seed, chi_to_b0=False, t2star="off")
        vox = torch.floor((ph.pos - torch.tensor(a.origin)) / torch.tensor(a.voxel).flip(0) + .5).long()
        lin = (vox[:, 2] * 4 + vox[:, 1]) * 4 + vox[:, 0]
        for i, name in enumerate(("wm", "gm", "csf")):
            sel = torch.isclose(ph.t1, ph.t1.new_tensor(DEFAULT_TISSUES[3.][name].t1.mean), rtol=0, atol=1e-12)
            count = torch.zeros(64, dtype=torch.float64).index_add_(0, lin[sel], torch.ones(int(sel.sum()), dtype=torch.float64))
            assert ((count - k * f[i].reshape(-1)).abs() < 1 + 1e-9).all()
        totals.append(float(ph.pd.sum()))
    exact = sum(float(f[i].sum()) * DEFAULT_TISSUES[3.][n].pd.mean for i, n in enumerate(("wm", "gm", "csf")))
    mean, se = np.mean(totals), np.std(totals) / np.sqrt(len(totals))
    assert abs(mean - exact) < 4 * se + 1e-12
    cat = [float(to_phantom(a, spins_per_voxel=(2, 2, 2), sampling="categorical", seed=s, chi_to_b0=False,
                            t2star="off").pd.sum()) for s in range(200)]
    assert np.std(totals) < 0.5 * np.std(cat)  # the point of it: far less draw noise


def test_layer_scalp_moves_the_subcutaneous_layer_into_fat():
    """A column of skin below air on a 0.5 mm grid: the membership between 1.61 mm and 5.75 mm below
    the surface (midway between the last air and first skin voxel centres) moves to fat, partial at
    the layer edges, exactly 4.14 mm of it; skin + fat is conserved, nothing else changes."""
    pytest.importorskip("scipy")
    n, h = 40, 0.5e-3
    air = torch.zeros(1, 1, n, dtype=torch.float64)
    air[..., :10] = 1
    skin = 0.9 * (1 - air)  # a fuzzy class: 10 % of each voxel is something else
    an = Anatomy({"background": air, "skin": skin, "fat": torch.zeros_like(air), "wm": 0.1 * (1 - air)},
                 (h, h, h), (0., 0., 0.))
    out = bp.layer_scalp(an)
    fat = out.fractions["fat"][0, 0]
    depth = (torch.arange(n, dtype=torch.float64) - 9.5) * h  # voxel centres below the surface
    assert float(fat.sum()) * h == pytest.approx(0.9 * 4.14e-3, rel=1e-12)
    assert torch.equal(out.fractions["skin"] + out.fractions["fat"], skin)
    assert torch.equal(out.fractions["wm"], an.fractions["wm"]) and torch.equal(out.fractions["background"], air)
    assert (fat[depth < 1.61e-3 - h / 2] == 0).all() and (fat[depth > 5.75e-3 + h / 2] == 0).all()
    assert torch.allclose(fat[(depth > 1.61e-3 + h / 2) & (depth < 5.75e-3 - h / 2)], torch.tensor(0.9, dtype=torch.float64))


def test_proportional_sampling_conserves_each_tissue_and_has_k_spins():
    """sampling="proportional": every voxel's PD per tissue exactly fraction * PD, max(1, round(K f))
    spins per tissue (about K per voxel)."""
    g = torch.Generator().manual_seed(0)
    shape = (2, 3, 4)
    a = torch.rand(shape, generator=g, dtype=torch.float64) * 0.7
    b = (1 - a) * torch.rand(shape, generator=g, dtype=torch.float64)
    an = Anatomy({"wm": a, "gm": b}, (1e-3,) * 3, (0., 0., 0.))
    ph = bp.to_phantom(an, DEFAULT_TISSUES, 3., spins_per_voxel=8, sampling="proportional", chi_to_b0=False, t2star="off")
    pd_wm, pd_gm = DEFAULT_TISSUES[3.]["wm"].pd.mean, DEFAULT_TISSUES[3.]["gm"].pd.mean
    t1_wm = DEFAULT_TISSUES[3.]["wm"].t1.mean
    is_wm = torch.isclose(ph.t1, torch.tensor(t1_wm, dtype=ph.t1.dtype))
    assert float(ph.pd[is_wm].sum()) == pytest.approx(pd_wm * float(a.sum()), rel=1e-12)
    assert float(ph.pd[~is_wm].sum()) == pytest.approx(pd_gm * float(b.sum()), rel=1e-12)
    expect = sum(int(max(1, round(8 * float(f)))) for f in torch.cat([a.ravel(), b.ravel()]) if f > 0)
    assert len(ph) == expect


def _two_banks(theta_deg=0., shape=(30, 48, 64), h=0.5e-3):
    """White matter on both sides of a 6 mm grey band (two touching cortical banks; its midline at
    x' = 12.75 mm, x' turned by theta in the (y, x) plane), CSF above z = 12 mm; far from MNI's deep grey."""
    nz, ny, nx = shape
    z, y, x = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) * h for n in shape), indexing="ij")
    t = math.radians(theta_deg)
    u = x * math.cos(t) + y * math.sin(t)
    below = z < 12e-3
    wm = (below & ((u < 9.75e-3) | (u > 15.75e-3))).double()
    gm = (below & (wm == 0)).double()
    an = Anatomy({"wm": wm, "gm": gm, "csf": (~below).double()}, (h,) * 3, (0.1, 0.1, 0.1))
    return an, u


def test_open_sulci_puts_csf_on_the_midline_between_two_banks():
    """open_sulci: width W of CSF goes onto the band's midline, W x the sheet's area of it (axis-aligned
    and turned 30 deg), taken from grey matter only; nothing is opened in a grey layer over a bumpy
    white surface (no second bank); width 0 changes nothing."""
    pytest.importorskip("scipy")
    W, h = 1.2e-3, 0.5e-3
    for theta in (0., 30.):
        an, u = _two_banks(theta)
        out = bp.open_sulci(an, W)
        added = out.fractions["csf"] - an.fractions["csf"]
        assert torch.allclose(out.fractions["gm"] + out.fractions["csf"], an.fractions["gm"] + an.fractions["csf"])
        assert torch.equal(out.fractions["wm"], an.fractions["wm"]) and float(added.min()) >= 0
        assert float(added[an.fractions["wm"] > 0].abs().sum()) == 0
        # inside a window away from the box edges: within W / 2 + h of the midline, W x its length x height
        nz, ny, nx = an.shape
        z, y, x = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) * h for n in an.shape), indexing="ij")
        win = (z >= 1e-3) & (z < 11e-3) & (y >= 3e-3) & (y < (ny - 6) * h) & (x >= 3e-3) & (x < (nx - 6) * h)
        assert float(added[win & ((u - 12.75e-3).abs() > W / 2 + h)].sum()) == 0
        s = torch.linspace(-0.1, 0.1, 400001, dtype=torch.float64)
        t = math.radians(theta)
        px, py = 12.75e-3 * math.cos(t) - s * math.sin(t), 12.75e-3 * math.sin(t) + s * math.cos(t)
        inside = (py >= 3e-3 - h / 2) & (py < (ny - 6) * h - h / 2) & (px >= 3e-3 - h / 2) & (px < (nx - 6) * h - h / 2)
        length = float(inside.sum()) * float(s[1] - s[0])
        assert float(added[win].sum()) * h ** 3 == pytest.approx(W * length * 10e-3, rel=0.1)
    # a 3 mm grey layer over white matter with 0.5 mm bumps, or 3 mm deep, 6 mm wide grooves: no sulcus
    shape = (30, 48, 64)
    z, y, x = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) * h for n in shape), indexing="ij")
    for top in (7e-3 + 0.5e-3 * torch.sin(2 * math.pi * x / 4e-3) * torch.sin(2 * math.pi * y / 4e-3),
                7e-3 + 1.5e-3 * torch.sin(2 * math.pi * x / 6e-3)):
        wm, gm = (z < top).double(), ((z >= top) & (z < 10e-3)).double()
        an = Anatomy({"wm": wm, "gm": gm, "csf": (z >= 10e-3).double()}, (h,) * 3, (0.1, 0.1, 0.1))
        assert float((bp.open_sulci(an, W).fractions["csf"] - an.fractions["csf"]).sum()) == 0
    assert bp.open_sulci(an, 0.) is an


def test_open_sulci_to_a_volume_and_gather_cortical_csf():
    """open_sulci(volume=V) adds V of CSF (the width solved by bisection), widen adds W x area on top;
    gather_cortical_csf gives each grey/CSF voxel to its larger share, white matter untouched."""
    pytest.importorskip("scipy")
    an, u = _two_banks(0.)
    h = 0.5e-3
    V = 150e-9  # m^3
    out = bp.open_sulci(an, volume=V)
    assert float((out.fractions["csf"] - an.fractions["csf"]).sum()) * h ** 3 == pytest.approx(V, rel=0.02)
    wide = bp.open_sulci(an, volume=V, widen=1e-3)
    area = 12e-3 * an.shape[1] * h  # the band's height x depth in y
    assert float((wide.fractions["csf"] - out.fractions["csf"]).sum()) * h ** 3 == pytest.approx(1e-3 * area, rel=0.1)
    with pytest.raises(ValueError):
        bp.open_sulci(an, 1e-3, volume=V)
    gm = torch.tensor([[[.7, .3, .5, 0., .6]]], dtype=torch.float64)
    csf = torch.tensor([[[.2, .6, .5, .4, 0.]]], dtype=torch.float64)
    wm = torch.tensor([[[.1, .1, 0., .6, .4]]], dtype=torch.float64)
    mixed = Anatomy({"gm": gm, "csf": csf, "wm": wm}, (h,) * 3, (0.1, 0.1, 0.1))
    g = bp.gather_cortical_csf(mixed)
    assert g.fractions["gm"].flatten().tolist() == pytest.approx([.9, 0., 0., 0., .6])
    assert g.fractions["csf"].flatten().tolist() == pytest.approx([0., .9, 1., .4, 0.])
    assert torch.equal(g.fractions["wm"], wm)


def test_sulcal_width_follows_jin_2018():
    """sulcal_width: Jin et al. 2018's means and SDs at their groups' mean ages, linear between."""
    assert bp.sulcal_width(47.2) == pytest.approx((1.27e-3, 0.17e-3))
    assert bp.sulcal_width(67.04) == pytest.approx((1.49e-3, 0.20e-3))
    assert bp.sulcal_width(25.)[0] < bp.sulcal_width(80.)[0]
    assert bp.sulcal_widening(25.) == 0 and bp.sulcal_widening(70.) == pytest.approx(40 * 0.22e-3 / 19.84)


def test_myelin_water_is_a_second_free_water_species():
    """myelin_water_fraction m: the water splits into (1 - m) at T2 and m at myelin_water_t2, total PD
    unchanged; None leaves one compartment (bit-identical spins)."""
    from blochparty.anatomy import Distribution
    a = torch.full((3, 3, 3), .6, dtype=torch.float64)
    an = Anatomy({"wm": a}, (1e-3,) * 3, (0., 0., 0.))
    base = DEFAULT_TISSUES[3.]["wm"]
    kw = dict(spins_per_voxel=1, chi_to_b0=False, t2star="off")
    one = bp.to_phantom(an, {3.: {"wm": base}}, 3., **kw)
    two = bp.to_phantom(an, {3.: {"wm": replace(base, myelin_water_fraction=Distribution(.12, None, "test"),
                                                myelin_water_t2=Distribution(.02, None, "test"))}}, 3., **kw)
    assert len(two) == 2 * len(one)
    assert float(two.pd.sum()) == pytest.approx(float(one.pd.sum()), rel=1e-12)
    short = torch.isclose(two.t2, torch.tensor(.02, dtype=two.t2.dtype))
    assert float(two.pd[short].sum()) == pytest.approx(.12 * float(one.pd.sum()), rel=1e-12)
    assert torch.allclose(two.t2[~short], one.t2)


def test_label_deep_grey_moves_grey_matter_inside_the_nuclei_only():
    """On a 2 mm grid in MNI coordinates: grey matter at each nucleus's centre (either side) moves to
    that nucleus, cortex-like positions (insula, cingulate, a frontal gyrus) stay gm, white matter and
    the total do not change; a second call refuses."""
    from blochparty.anatomy import DEEP_GREY
    h = 2e-3
    shape = (31, 51, 51)  # z -20..40, y -60..40, x -50..50 mm
    gm = torch.full(shape, .7, dtype=torch.float64)
    an = Anatomy({"gm": gm, "wm": torch.full(shape, .3, dtype=torch.float64)}, (h, h, h), (-.05, -.06, -.02))
    out = bp.label_deep_grey(an)
    fr = out.fractions
    assert torch.equal(fr["wm"], an.fractions["wm"])
    assert torch.allclose(sum(fr[k] for k in ("gm", *DEEP_GREY)), gm, rtol=0, atol=1e-15)

    def at(x, y, z):  # mm -> (z, y, x) index
        return round((z + 20) / 2), round((y + 60) / 2), round((x + 50) / 2)

    for name, regions in DEEP_GREY.items():
        (cx, cy, cz), _ = regions[0]
        for x in (-cx, cx):
            assert float(fr[name][at(x, cy, cz)]) == .7 and float(fr["gm"][at(x, cy, cz)]) == 0
    for x, y, z in ((40, 0, 0), (4, 20, 30), (30, 36, 10)):
        assert float(fr["gm"][at(x, y, z)]) == .7
        assert all(float(fr[k][at(x, y, z)]) == 0 for k in DEEP_GREY)
    with pytest.raises(ValueError):
        bp.label_deep_grey(out)


def test_ventricle_displacement_grows_each_ventricle_and_keeps_the_septum():
    """Two CSF balls (radius 6 mm) either side of x = 0, 2 mm of white matter between them, on a 0.5 mm
    grid: a 2 mm wall offset grows each ball's CSF to about (8/6)^3 of its volume (the pull-back
    copies the interior outward; trilinear edges), leaves the septum white matter and loses tissue
    where the walls moved."""
    from blochparty.anatomy import ventricle_displacement
    h, n = .5e-3, 81
    g = torch.arange(n, dtype=torch.float64) * h - 20e-3
    z, y, x = torch.meshgrid(g, g, g, indexing="ij")
    csf = ((torch.sqrt(z ** 2 + y ** 2 + (x - 7e-3) ** 2) < 6e-3) | (torch.sqrt(z ** 2 + y ** 2 + (x + 7e-3) ** 2) < 6e-3)).double()
    an = Anatomy({"csf": csf, "wm": 1 - csf}, (h,) * 3, (-20e-3,) * 3)
    out = bp.deform(an, ventricle_displacement(an, csf.numpy() > .5, 2e-3))
    assert float(out.fractions["csf"].sum()) == pytest.approx(float(csf.sum()) * (8 / 6) ** 3, rel=.12)
    assert float(out.fractions["wm"][40, 40, 40]) == pytest.approx(1., abs=1e-6)  # the septum at x = 0
    assert float(out.fractions["wm"].sum()) < float(an.fractions["wm"].sum())


def test_add_falx_splits_the_fissure_above_the_callosum():
    """Two grey-matter hemispheres around a 3 mm CSF fissure at x = 0, a callosum (midline white matter) at z = 0..3 mm
    with CSF under it (a ventricle): the falx (dura) goes into the fissure only more than FALX_ABOVE_CALLOSUM above
    the callosum, FALX_THICKNESS per row, taken from the CSF; none under the callosum."""
    h = 0.5e-3
    n = (80, 8, 41)  # z, y, x
    x = (np.arange(n[2]) - 20) * h
    z = (np.arange(n[0]) - 20) * h
    csf = np.zeros(n, np.float32); gm = np.zeros(n, np.float32); wm = np.zeros(n, np.float32)
    fissure = np.abs(x) <= 1.5e-3
    csf[:, :, fissure] = 1
    gm[:, :, ~fissure] = 1
    cc = (z >= 0) & (z < 3e-3)
    csf[cc] = 0; gm[cc] = 0; wm[cc] = 1
    an = bp.Anatomy({"csf": torch.from_numpy(csf), "gm": torch.from_numpy(gm), "wm": torch.from_numpy(wm),
                     "dura": torch.zeros(n)}, voxel=(h,) * 3, origin=(x[0], 0.0, z[0]))
    out = bp.add_falx(an)
    dura = out.fractions["dura"].numpy()
    total = out.fractions["csf"].numpy() + dura
    assert np.allclose(total, csf)
    per_row = dura.sum(-1) * h  # (z, y) [m]
    top = z[cc].max()
    high = z > top + bp.anatomy.FALX_ABOVE_CALLOSUM + h
    assert np.allclose(per_row[high], bp.anatomy.FALX_THICKNESS, rtol=1e-5)
    assert per_row[(z > top) & (z <= top + bp.anatomy.FALX_ABOVE_CALLOSUM)].max() == 0
    assert per_row[z < 0].max() == 0
    # centred on the fissure
    assert abs((dura[high] * x).sum() / dura[high].sum()) < 1e-6


def test_deform_cubic_keeps_fine_structure():
    """A 3-voxel-period pattern shifted by half a voxel: the cubic B-spline pull-back (order 3) reproduces it far better
    than trilinear; with no displacement it returns the fractions unchanged; fractions still sum to at most one."""
    n = (4, 6, 48)
    xi = np.arange(n[2])
    wave = (0.5 + 0.4 * np.sin(2 * np.pi * xi / 3.0)).astype(np.float32)
    a = torch.from_numpy(np.broadcast_to(wave, n).copy())
    an = bp.Anatomy({"gm": a, "wm": 1 - a}, voxel=(1e-3,) * 3, origin=(0.0, 0.0, 0.0))
    still = bp.deform(an, torch.zeros((3, *n)), order=3)
    assert torch.allclose(still.fractions["gm"], a, atol=1e-5)
    shift = torch.zeros((3, *n)); shift[2] = 0.5e-3
    truth = 0.5 + 0.4 * np.sin(2 * np.pi * (xi + 0.5) / 3.0)
    inner = slice(6, -6)
    err = {o: np.abs(bp.deform(an, shift, order=o).fractions["gm"].numpy()[1, 2, inner] - truth[inner]).max() for o in (1, 3)}
    assert err[3] < 0.35 * err[1]
    out = bp.deform(an, shift, order=3)
    assert float((out.fractions["gm"] + out.fractions["wm"]).max()) <= 1 + 1e-6


def test_residual_scale_maps_multiply_non_fat_tissues():
    """t2_scale / pd_scale / t1_scale maps multiply each non-fat tissue's T2, PD, T1 per voxel; fat keeps its values;
    without the maps nothing changes."""
    from blochparty.anatomy import DEFAULT_TISSUES
    fr = {"wm": torch.tensor([[[.6, 1.]]], dtype=torch.float64), "fat": torch.tensor([[[.4, 0.]]], dtype=torch.float64)}
    base = Anatomy(fr, (.001,) * 3, (0, 0, 0))
    scales = {"t2_scale": torch.tensor([[[1.3, .8]]], dtype=torch.float64),
              "pd_scale": torch.tensor([[[1.1, .9]]], dtype=torch.float64),
              "t1_scale": torch.tensor([[[1.2, 1.]]], dtype=torch.float64)}
    a = Anatomy(fr, (.001,) * 3, (0, 0, 0), maps=scales)
    kw = dict(chi_to_b0=False, spins_per_voxel=1, jitter=None, t2star="off")
    p0, p1 = to_phantom(base, **kw), to_phantom(a, **kw)
    wm = DEFAULT_TISSUES[3.]["wm"]
    x0 = p0.pos[:, 2] < .0005  # the first voxel's spins (x index 0)
    is_wm0 = x0 & torch.isclose(p0.t2, torch.tensor(wm.t2.mean, dtype=p0.t2.dtype))
    assert is_wm0.any()
    i = is_wm0.nonzero()[0, 0]
    assert float(p1.t2[i]) == pytest.approx(1.3 * wm.t2.mean)
    assert float(p1.t1[i]) == pytest.approx(1.2 * wm.t1.mean)
    assert float(p1.pd[i]) == pytest.approx(1.1 * float(p0.pd[i]))
    fat0 = x0 & ~torch.isclose(p0.t2, torch.tensor(wm.t2.mean, dtype=p0.t2.dtype))
    assert torch.allclose(p1.t2[fat0], p0.t2[fat0]) and torch.allclose(p1.pd[fat0], p0.pd[fat0])


def test_add_dermis_is_a_conserved_surface_shell():
    """A sphere of skin (R 40 mm, 0.5 mm voxels): the dermis takes the outer shell (volume ~ 4 pi R^2 t for
    t = 1.61 mm, within a voxel's rounding), nothing deeper than t + a voxel, every voxel's total unchanged."""
    from blochparty.anatomy import SCALP_LAYERS
    h, n = 0.5e-3, 180
    c = (np.arange(n) - n / 2 + .5) * h
    zz, yy, xx = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
    skin = torch.from_numpy(np.clip((40e-3 - r) / h + .5, 0, 1).astype(np.float64))
    a = bp.Anatomy({"skin": skin, "background": 1 - skin}, (h,) * 3, (0, 0, 0))
    d = bp.add_dermis(a)
    t = SCALP_LAYERS[0]
    vol = float(d.fractions["dermis"].sum()) * h ** 3
    assert vol == pytest.approx(4 * math.pi * 40e-3 ** 2 * t, rel=0.05)
    total = d.fractions["skin"] + d.fractions["dermis"] + d.fractions["background"]
    assert torch.allclose(total, torch.ones_like(total))
    assert float(d.fractions["dermis"][torch.from_numpy(r < 40e-3 - t - 2 * h)].max()) == 0


def test_add_lesions_honours_count_size_and_conservation():
    """A white-matter block with a cortex shell: the requested number of lesions appears, each lesion's placed volume
    is within the ellipsoid range the size bounds allow, lesion mass comes only from white matter (grey matter and
    CSF untouched), every voxel's total stays 1, and the draw is reproducible from its generator."""
    from blochparty.anatomy import add_lesions
    h, n = 1e-3, 60
    c = (np.arange(n) - n / 2 + .5) * h
    zz, yy, xx = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
    wm = torch.from_numpy((r < 22e-3).astype(np.float64))
    gm = torch.from_numpy(((r >= 22e-3) & (r < 26e-3)).astype(np.float64))
    csf = torch.from_numpy(((r >= 26e-3) & (r < 28e-3)).astype(np.float64))
    bg = 1 - wm - gm - csf
    a = bp.Anatomy({"wm": wm, "gm": gm, "csf": csf, "background": bg}, (h,) * 3, (-n / 2 * h,) * 3)
    out, records = add_lesions(a, np.random.default_rng(3), count=4, size_mm=(4., 8.), region="any", spacing_mm=12.)
    assert len(records) == 4
    total = sum(out.fractions.values())
    assert torch.allclose(total, torch.ones_like(total))
    assert torch.equal(out.fractions["gm"], gm) and torch.equal(out.fractions["csf"], csf)
    assert torch.allclose(out.fractions["wm"] + out.fractions["lesion"], wm)
    for rec in records:
        lo, hi = (4 / 3 * math.pi * (2e-3 * .5) ** 2 * 2e-3 * 1e6, 4 / 3 * math.pi * (4e-3) ** 3 * 1e6)
        assert lo * 0.5 <= rec["volume_ml"] <= hi * 1.5, rec  # soft edges widen the range a little
        assert 2. <= rec["semi_axes_mm"][0] <= 4. and all(1. <= s <= 4. for s in rec["semi_axes_mm"][1:])
        assert rec["semi_axes_mm"][0] == max(rec["semi_axes_mm"])
    again, rec2 = add_lesions(a, np.random.default_rng(3), count=4, size_mm=(4., 8.), region="any", spacing_mm=12.)
    assert torch.equal(again.fractions["lesion"], out.fractions["lesion"]) and rec2 == records
    # juxtacortical lesions sit within 3 mm of the cortex; deep ones need a ventricle (MNI) frame: not here
    out2, rec3 = add_lesions(a, np.random.default_rng(1), count=2, size_mm=(3., 5.), region="juxtacortical")
    for rec in rec3:
        assert np.linalg.norm(np.array(rec["centre_mm"]) * 1e-3) >= 22e-3 - 3e-3 - 1e-3
    with pytest.raises(ValueError):
        add_lesions(a, np.random.default_rng(0), region="cortical")
