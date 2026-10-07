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
    no_micro = to_phantom(a,field_maps=pair)
    assert not no_micro.db0.any()
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
    assert DEFAULT_TISSUES[3]["csf"].t1.assumed
    assert DEFAULT_TISSUES[3]["fat"].t2.assumed
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
    with pytest.raises(NotImplementedError):
        to_phantom(a,variability={})
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
