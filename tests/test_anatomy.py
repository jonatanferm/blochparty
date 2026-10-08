"""Anatomy assembly vs conservation, Ernst, continuous-box FT and sphere fields.

All unit fixtures are synthetic; real BrainWeb checks and throughput live in
validation/anatomy_brainweb.py and validation/anatomy/RESULTS.md.
"""
from dataclasses import replace, fields
import gzip
import hashlib
import json
import math

import numpy as np
import pytest
import torch

import blochparty as bp
from blochparty.anatomy import (AIR_CHI, Anatomy, BRAINWEB_TISSUES, DEFAULT_TISSUES,
    Distribution, TissueProperties, anatomy_field_maps, load_brainweb, to_phantom)
from blochparty.maps import sample
from blochparty.phantom import b0_sphere
from test_susceptibility import sphere


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


def test_gpu_equals_core():
    metal = pytest.importorskip("blochparty.metal")
    if not metal.available():
        pytest.skip("no Apple GPU")
    a = mixed((3,3,3))
    ph = to_phantom(a,mt=True,diffusion=True,spins_per_voxel=8)
    q = bp.SequenceBuilder().rf(.001,flip_deg=35,n_steps=50).readout(12,.0002).build()
    ref = bp.simulate(ph,q).signal
    actual = metal.simulate_metal(ph,q).signal
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
