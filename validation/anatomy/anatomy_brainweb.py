"""BrainWeb contrast plausibility and full-head throughput (not scanner ground truth).

    uv run python validation/anatomy/fetch_brainweb.py --legacy
    uv run python validation/anatomy/anatomy_brainweb.py [--benchmark]

The 20-model set supplies only SFLASH T1 (22/9.2 ms, 30 deg). The older
single-subject SBD supplies T1 (18/10 ms, 30 deg) and double-spin-echo T2/PD
(TR=3300 ms, echoes 35/120 ms). Templates were inspected on 2026-10-07 at
https://brainweb.bic.mni.mcgill.ca/cgi/bw/submit_request . No custom job is sent.
Compare pure-tissue (>95%) medians in downloaded images to Bloch tissue signals.
With uniform fields, noise-free pure-tissue signal ratios do not require a full
k-space reconstruction. The tissue signals also linearly synthesize mixtures.
Published in-vivo T1/T2 values are propagated through identical protocols using
Wansapura et al. (1999, PMID:10232510) as a human 3 T comparison, not as a 1.5 T
ratio reference: field strength and population differ. No universal in-vivo
contrast ratio is claimed. See RESULTS.md for that limitation.
"""
import argparse
from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch

import blochparty as bp
from blochparty.anatomy import (Anatomy, BRAINWEB_TISSUES, DEFAULT_TISSUES, Distribution,
                               anatomy_field_maps, load_brainweb, to_phantom)

ROOT = Path(__file__).parent/"data"


def read(entry,shape):
    raw = gzip.open(ROOT/entry["path"],"rb").read()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("reference checksum mismatch")
    a = np.frombuffer(raw,dtype="<u2").reshape(shape)
    if a.max() > 4095:
        raise ValueError("reference byte order/range mismatch")
    return a


def tissue_signals(table,strength,tr,echoes,flip=30):
    names = ["wm","gm","csf"]
    fractions = {name:torch.nn.functional.one_hot(torch.tensor(i),3).reshape(1,1,3).double()
                 for i,name in enumerate(names)}
    a = Anatomy(fractions,(.001,)*3,(0,0,0))
    k = 4096 if len(echoes) == 1 else 8
    ph = to_phantom(a,table,strength,spins_per_voxel=(1,1,k),jitter=None,chi_to_b0=False,t2star="table")
    ids = torch.round(ph.pos[:,0]/.001).long()
    coils = torch.nn.functional.one_hot(ids,3).to(torch.complex128)
    b = bp.SequenceBuilder()
    pulse,dwell = 1e-9,1e-9
    if len(echoes) == 1:
        e = torch.exp(-tr/ph.t1)
        initial = torch.zeros_like(ph.pos)
        initial[:,2] = ph.pd*(1-e)/(1-e*math.cos(math.radians(flip)))
        b.rf(pulse,flip_deg=flip,n_steps=1).delay(echoes[0]-pulse).readout(1,dwell)
        s = bp.simulate(ph,b.build(),coils,m_init=initial).signal.abs()
    else:
        # Repeated ideal DSE: steady state includes longitudinal recovery and
        # inversions at BOTH refocusing pulses, rather than a guessed SE formula.
        first,second = echoes
        for _ in range(24):
            b.rf(pulse,flip_deg=90,n_steps=1).delay(first/2-pulse)
            b.rf(pulse,flip_deg=180,phase=math.pi/2,n_steps=1).delay(first/2-pulse)
            b.readout(1,dwell).delay((second-first)/2-dwell)
            b.rf(pulse,flip_deg=180,phase=math.pi/2,n_steps=1).delay((second-first)/2-pulse)
            b.readout(1,dwell).delay(tr-second-dwell).spoil()
        s = bp.simulate(ph,b.build(),coils).signal[:,-2:].abs()
    return s.numpy()


def ratios(values):
    return {"GM/WM":float(values[1]/values[0]),"CSF/GM":float(values[2]/values[1])}


def contrast():
    legacy = json.loads((ROOT/"legacy.json").read_text())
    masks = [read(legacy["files"][name],legacy["shape"]) > .95*4095 for name in ("wm","gm","csf")]
    sim_t1 = tissue_signals(BRAINWEB_TISSUES,1.5,.018,[.010])[:,0]
    sim_dse = tissue_signals(BRAINWEB_TISSUES,1.5,3.3,[.035,.120])
    results = {}
    for protocol,signal in (("T1",sim_t1),("T2",sim_dse[:,1]),("PD",sim_dse[:,0])):
        image = read(legacy["files"][protocol],legacy["shape"])
        reference = np.array([np.median(image[m]) for m in masks])
        a,b = ratios(signal),ratios(reference)
        results[protocol] = dict(bloch=a,brainweb=b,
            relative_difference={k:a[k]/b[k]-1 for k in a},roi_voxels=[int(m.sum()) for m in masks])
    # Independent human relaxation measurements, without pretending these are
    # observed clinical image ratios or matched 1.5 T measurements.
    # The abstract's "T2 ... gray matter and white matter were 80 and 110 msec" is in the
    # order opposite to every other 3 T brain study (GM T2 > WM T2; Stanisz 2005: 99 / 69 ms),
    # so the T2 values are assigned by physics; the full text could not be checked.
    human = dict(DEFAULT_TISSUES[3.])
    source = ("Wansapura et al. 1999, abstract, in-vivo 3 T mean, T2 GM/WM order swapped "
              "from the abstract by physics (full text not read); PMID:10232510")
    for name,t1,t2 in (("wm",.832,.080),("gm",1.331,.110)):
        human[name] = replace(human[name],t1=Distribution(t1,None,source),t2=Distribution(t2,None,source))
    for protocol,signal in (("T1",tissue_signals({3.:human},3.,.018,[.010])[:,0]),
                            ("T2",tissue_signals({3.:human},3.,3.3,[.035,.120])[:,1]),
                            ("PD",tissue_signals({3.:human},3.,3.3,[.035,.120])[:,0])):
        results[protocol]["human_3T_parameters_GM/WM"] = ratios(signal)["GM/WM"]
    # 20-model T1 reference has a larger transverse FOV and a quarter-mm offset
    # from 1-mm block-averaged fuzzy models: sample reference at anatomy centres.
    a = load_brainweb(4,ROOT,voxel=(.001,)*3)
    manifest = json.loads((ROOT/"subject04.json").read_text())
    image = torch.from_numpy(read(manifest["files"]["t1w"],(181,256,256)).astype(np.float32))
    medians = []
    from blochparty.maps import sample
    for name in ("wm","gm","csf"):
        idx = (a.fractions[name] > .95).nonzero().float()
        pos = idx.flip(-1)*.001+torch.tensor(a.origin)
        values = sample(image[None],torch.tensor([-.12775,-.14575,-.07225]),torch.tensor([.001]*3),pos)[:,0]
        medians.append(values.median().item())
    signal = tissue_signals(BRAINWEB_TISSUES,1.5,.022,[.0092])[:,0]
    results["subject04_T1"] = dict(bloch=ratios(signal),brainweb=ratios(np.array(medians)))
    # Susceptibility sources are the complete head, not a cropped brain/slab.
    coarse = a.resample((.002,)*3)
    tissue,scanner = anatomy_field_maps(coarse)
    mask = sum(coarse.fractions[n] for n in ("wm","gm","csf")) > .5
    before,after = tissue.b0[mask],(tissue.b0+scanner.b0)[mask]
    results["subject04_B0_2mm"] = dict(brain_rms_before_Hz=float(before.square().mean().sqrt()),
        brain_rms_after_Hz=float(after.square().mean().sqrt()),
        residual_percentiles_Hz=torch.quantile(after,torch.tensor([.01,.5,.99])).tolist(),
        note="scalar water-like tissue/air and -2 ppm bone proxy; plausibility, not a measured field comparison")
    return results


def benchmark():
    if not bp.engines.available("gpu"):
        raise RuntimeError("no GPU engine here (Apple GPU or NVIDIA); run with host GPU access")
    started = time.perf_counter()
    a = load_brainweb(4,ROOT,voxel=(.001,)*3)
    load_seconds = time.perf_counter()-started
    started = time.perf_counter()
    # Full head, one sampled tissue per anatomy voxel, air dropped. This measures
    # volumetric throughput, not spatial or T2' convergence. Fat splits into six peaks.
    ph = to_phantom(a,spins_per_voxel=1,sampling="categorical",chi_to_b0=False,t2star="off")
    assembly_seconds = time.perf_counter()-started
    sequence = bp.SequenceBuilder(dtype=torch.float32).rf(.001,flip_deg=30,n_steps=50).readout(32,.0002).build()
    small = replace(ph,**{key:getattr(ph,key)[:100] for key in ("pos","pd","t1","t2","db0")})
    bp.simulate(small,sequence,engine="gpu")  # shader warm-up excluded
    started = time.perf_counter()
    out = bp.simulate(ph,sequence,engine="gpu")  # results come back on the CPU: the GPU has finished
    elapsed = time.perf_counter()-started
    return dict(spins=len(ph),load_seconds=load_seconds,assembly_seconds=assembly_seconds,
        gpu_seconds=elapsed,spins_per_second=len(ph)/elapsed,segments=len(sequence),adc_samples=32,
        phantom_bytes=sum(getattr(ph,k).numel()*getattr(ph,k).element_size() for k in ("pos","pd","t1","t2","db0")),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),finite_signal=bool(torch.isfinite(out.signal).all()),
        note="full 1-mm head; categorical K=1; no susceptibility/T2prime/MT/diffusion; RF+FID throughput, not imaging speed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--benchmark-only",action="store_true")
    args = parser.parse_args()
    results = {} if args.benchmark_only else contrast()
    if args.benchmark or args.benchmark_only:
        results["benchmark"] = benchmark()
    print(json.dumps(results,indent=2))
    output = "benchmark_results.json" if args.benchmark_only else "validation_results.json"
    (ROOT/output).write_text(json.dumps(results,indent=2)+"\n")
