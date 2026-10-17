"""The tissue tables with their citations (``DEFAULT_TISSUES``, ``BRAINWEB_TISSUES``), magnetization transfer
and diffusion per tissue."""

from __future__ import annotations

from dataclasses import replace

from .grid import AIR_CHI, WATER_CHI, Distribution, TissueProperties

# Every fallback explicitly says it is an assumption. These are simulation priors,
# not a normative human atlas. No values in phantom.TISSUES are changed.
_BW = "BrainWeb MRISIM tissue_mr_parameters.txt (1996), named tissue row; https://brainweb.bic.mni.mcgill.ca/brainweb/tissue_mr_parameters.txt"
_ST = "Stanisz et al. 2005, Table 1, doi:10.1002/mrm.20605; 37 C excised animal tissue (human blood)"
_CA = "Caan et al. 2019, Table 1 MP2RAGEME column, doi:10.1002/hbm.24490; human 7 T"
_RO = ("Rooney et al. 2007, doi:10.1002/mrm.21122: CSF T1 ~4.4 s, nearly field-independent (value as quoted by "
       "Spijkerman et al. 2018, ref. 13)")
_SP = ("Spijkerman et al. 2018, doi:10.1007/s10334-017-0659-3: 3 T ventricular CSF T2 median 2.03 s "
       "(range 1.73-2.16 s; peripheral 1.79 s), T2-prepared, 7 volunteers")
_SH = ("Shin et al. 2025, Table 3 average across 4 subjects (mean +- between-subject SD), doi:10.1002/mrm.30624: "
       "human 3 T (Prisma), FreeSurfer GM/WM, 1 x 1 x 2 mm, TE 3-66.18 ms (43 echoes), mono-exponential, no "
       "macroscopic-field correction; the abstract swaps the WM/GM labels")
_DB = "de Bazelaire et al. 2004, doi:10.1148/radiol.2303021331: subcutaneous fat at 3 T, abdomen"
_IT = ("IT'IS Foundation tissue database, relaxation times, SAT (subcutaneous fat) 1.5 T: mean of 2 studies "
       "(T1 288-343 ms, T2 58-165 ms); itis.swiss, checked 2026-10-07")
_SO = ("Song, Wehrli and Ma 1997, doi:10.1002/mrm.1910370207: in vivo skin MR microscopy at 1.5 T; the dermis's "
       "major transverse component (91 %) T2* ~10 ms")
_RI = ("Richard et al. 1991 / 1993, doi:10.1111/1523-1747.ep12478540 / 10.1111/1523-1747.ep12472356: skin layers at "
       "1.5 T have shorter T2 than other soft tissues (fibrous protein), T1 in the range of other tissues; the "
       "epidermis's mobile water is at least twice the dermis's")
_ET = ("Ettl et al. 1994, doi:10.1159/000310486: vitreous of 8 normal eyes at 1.5 T, T1 3257 +- 307 ms, T2 546 +- "
       "157 ms")
_WEST = ("West et al. 2014, doi:10.1371/journal.pone.0095161: MS lesions at 1.5 T by QRAPMASTER (synthetic MRI), R1 1.09 "
       "+- 0.11 1/s, R2 7.97 +- 0.91 1/s, PD 82.2 +- 3.8 % (controls' white matter 1.65, 12.87, 65.8 %)")
_SM24 = ("Simpson et al. 2024, doi:10.1002/mrm.29849: vitreous T1 at 3 T by inversion-recovery TrueFISP, healthy "
         "control eyes of 8 patients, 4.518 s")


def _assumed(value, reason):
    return Distribution(value, None, "Assumption: " + reason, True)


def _tables():
    # BrainWeb's legacy simulator values are useful for reproducibility only.
    legacy = dict(csf=(1., 2.569, .329, .058), gm=(.86, .833, .083, .069),
                  wm=(.77, .500, .070, .061), fat=(1., .350, .070, .058),
                  muscle=(1., .900, .047, .030))
    legacy |= {"skin": legacy["muscle"], "connective": legacy["muscle"],
               "dura": legacy["muscle"], "marrow": legacy["fat"],
               "vessels": legacy["gm"], "skull": (0., 1., .001, .001),
               "background": (0., 1., 1., 1.)}
    bw = {}
    for name, values in legacy.items():
        assumed = name not in ("csf", "gm", "wm", "fat", "muscle")
        params = [Distribution(v, None, _BW + f"; {name}" +
                   ("; proxy/placeholder for new tissue, not a measured value" if assumed else ""), assumed)
                  for v in values]
        chi = AIR_CHI if name == "background" else WATER_CHI
        if name == "skull":
            chi -= 2e-6  # M1's cited scale; explicit approximate contrast, not UTE calibration.
        bw[name] = TissueProperties(params[0], params[1], params[2],
            _assumed(chi, "water-like soft tissue; bone -2 ppm relative to water; air +0.36 ppm; see susceptibility.py"),
            params[3], fat_fraction=_assumed(float(name in ("fat", "marrow")), "pure lipid class or no lipid"))
    # A published simulator constant is not a measured human property. Flag
    # every legacy fallback, especially when carrying 1.5 T priors to 3 T.
    default = {}
    for strength in (1.5, 3.):
        default[strength] = {}
        for name, p in bw.items():
            fallback = {key: _assumed(getattr(p, key).mean,
                        getattr(p, key).citation + f"; legacy 1.5 T simulator prior used at {strength} T")
                        for key in ("pd", "t1", "t2", "t2star")}
            default[strength][name] = replace(p, **fallback)
    # 1.5 T / 3 T means and specimen SDs, seconds. In-vitro proxies are flagged.
    rows = {"wm": ((.884,.050,.072,.004), (1.084,.045,.069,.003)),
            "gm": ((1.124,.050,.095,.008), (1.820,.114,.099,.007)),
            "muscle": ((1.008,.020,.044,.006), (1.412,.013,.050,.004)),
            "vessels": ((1.441,.120,.290,.030), (1.932,.085,.275,.050))}
    for strength, index in ((1.5, 0), (3., 1)):
        for name, measurements in rows.items():
            t1, s1, t2, s2 = measurements[index]
            # T2* at 1.5 T is a legacy model prior; at 3 T WM/GM are whole-brain measurements
            # (they replace occipital Wansapura 1999 values, GM 41.6 < WM 48.4 ms, which made cortex
            # dark at long TE; Peters et al., ISMRM 2006 #926, also find GM 59.7 > WM 54.6 ms).
            star = min(bw[name].t2star.mean, t2)
            star_dist = _assumed(star, _BW + "; T2* proxy, clipped to T2; no measured SD")
            if strength == 3 and name in ("wm", "gm"):
                star_dist = Distribution(.0515, .0022, _SH) if name == "wm" else Distribution(.0559, .0018, _SH)
            default[strength][name] = replace(default[strength][name],
                t1=Distribution(t1,s1,_ST + f"; {name}, {strength} T",True),
                t2=Distribution(t2,s2,_ST + f"; {name}, {strength} T",True), t2star=star_dist)
        for name in ("skin", "connective", "dura"):
            proxy = default[strength]["muscle"]
            default[strength][name] = replace(proxy,
                t1=_assumed(proxy.t1.mean, _ST + f"; muscle used for {name}"),
                t2=_assumed(proxy.t2.mean, _ST + f"; muscle used for {name}"))
        # the dermis as its own class (anatomy.add_dermis): short T2 from its collagen, low mobile water
        muscle = default[strength]["muscle"]
        default[strength]["dermis"] = replace(muscle,
            pd=_assumed(.6, _RI + "; relative mobile-water PD 0.6 assumed (below soft tissue's)"),
            t1=_assumed(muscle.t1.mean, _RI + f"; muscle's {strength} T T1 used"),
            t2=_assumed(.012, _SO + "; T2 >= T2*: 12 ms assumed at 1.5 and 3 T"),
            t2star=_assumed(.010, _SO + (f"; 1.5 T value used at {strength} T" if strength != 1.5 else "")))
        # the vitreous (class eye): nearly pure water in a dilute collagen / hyaluronan network
        eye_t1 = (Distribution(3.257, .307, _ET) if strength == 1.5 else Distribution(4.518, None, _SM24)
                  if strength == 3. else _assumed(4.518, _SM24 + f"; used at {strength} T"))
        eye_t2 = Distribution(.546, .157, _ET) if strength == 1.5 else _assumed(.546, _ET + f"; 1.5 T T2 used at {strength} T")
        default[strength]["eye"] = replace(default[strength]["csf"], t1=eye_t1, t2=eye_t2,
            pd=_assumed(1., "vitreous ~99 % water: PD of CSF"),
            t2star=_assumed(eye_t2.mean, "vitreous T2* = T2: no intrinsic broadening assumed"))
        # white-matter lesions (class lesion, anatomy.add_lesions): MS plaques measured at 1.5 T (West 2014, SyMRI);
        # at 3 T the white matter's values scaled by the measured ratios (T1 1.51 x, T2 1.61 x), assumed
        wm_ = default[strength]["wm"]
        if strength == 1.5:
            default[strength]["lesion"] = replace(wm_, pd=Distribution(.822, .038, _WEST), t1=Distribution(.917, .093, _WEST),
                                                  t2=Distribution(.125, .014, _WEST), t2star=_assumed(.1, _WEST + "; T2* 0.8 x T2 assumed"))
        else:
            default[strength]["lesion"] = replace(wm_, pd=Distribution(.822, .038, _WEST),
                t1=_assumed(wm_.t1.mean * 1.51, _WEST + f"; the 1.5 T lesion / WM T1 ratio 1.51 on this table's {strength} T WM"),
                t2=_assumed(wm_.t2.mean * 1.61, _WEST + f"; the 1.5 T lesion / WM T2 ratio 1.61 on this table's {strength} T WM"),
                t2star=_assumed(wm_.t2.mean * 1.61 * .8, _WEST + "; T2* 0.8 x T2 assumed"))
        # CSF and subcutaneous fat: measured values replace the legacy 1.5 T simulator priors,
        # whose CSF (T1 2.57 s, T2 0.33 s) was the largest error in T2-weighted contrast.
        csf, fat = default[strength]["csf"], default[strength]["fat"]
        csf_t2 = Distribution(2.03, None, _SP, strength != 3.)
        default[strength]["csf"] = replace(csf, t1=Distribution(4.4, None, _RO),
            t2=csf_t2 if strength == 3. else replace(csf_t2, citation="Assumption: " + _SP + "; 3 T value used at 1.5 T"),
            t2star=_assumed(2.03, "CSF T2* = T2: no intrinsic broadening; pulsation and macroscopic fields not included"))
        if strength == 3.:
            default[strength]["fat"] = replace(fat, t1=Distribution(.382, .013, _DB), t2=Distribution(.068, .004, _DB))
        else:
            default[strength]["fat"] = replace(fat, t1=Distribution(.3124, .029, _IT), t2=Distribution(.1174, .0564, _IT),
                t2star=_assumed(.058, _BW + "; fat T2* legacy prior"))
        for name, values in _mt_and_diffusion(strength).items():
            default[strength][name] = replace(default[strength][name], **values)
    # Only verified 7 T brain entries: the remaining classes require user-supplied
    # values. T2 is an explicit 3 T proxy; the T1/T2* means AND SDs are measured.
    default[7.] = {}
    for name, t1, s1, star, ss in (("wm",1.19,.02,.0267,.0001), ("gm",2.,.08,.0324,.0007)):
        p = replace(default[3.][name], **_mt_and_diffusion(7.)[name])
        default[7.][name] = replace(p, t1=Distribution(t1,s1,_CA), t2star=Distribution(star,ss,_CA),
            t2=_assumed(p.t2.mean, _ST + "; 3 T T2 carried to 7 T pending calibrated value"))
    return default, {1.5: bw}


# Two-pool MT and isotropic diffusion (ROADMAP M5). MT in BoundPool's convention: bound_fraction
# p = M0b / (M0b + M0f), exchange_rate = bound -> free [1/s] (exchange.py), super-Lorentzian T2b.
_YA16 = ("Yarnykh 2016, doi:10.1002/mrm.25811: 3 T single-point MPF = M0b/(M0b+M0f), whole-brain WM and GM of "
         "8 healthy subjects, mean +- between-subject SD")
_YA12 = ("Yarnykh 2012, doi:10.1002/mrm.23224, Eq. 2 and 7: 3 T multi-point pulsed qMT, whole-brain histogram "
         "medians of 2 controls and 2 MS patients; R = k(1-f)/f is the bound-to-free rate; super-Lorentzian T2B")
_SI = ("Sinclair et al. 2010, doi:10.1002/mrm.22562: in vivo qMT of human calf muscle at 3 T, soleus, 10 "
       "subjects, mean +- SD; f of the total pool, super-Lorentzian T2b")
_HE = ("Helenius et al. 2002, AJNR 23:194 (www.ajnr.org/content/23/2/194): ADC at b = 0/1000 s/mm^2, 80 adults "
       "at 1.5 T, mean +- SD; diffusivity does not depend on B0")
_HO = ("Holz, Heil and Sacco 2000, doi:10.1039/b005319h: water self-diffusion, Speedy-Angell fit "
       "D0 (T/215.05 K - 1)^2.063, D0 = 1.635e-8 m^2/s, at 37 C (2.299e-9 at 25 C); in vivo CSF adds pulsation")
_LI = ("Li et al. 2014, doi:10.1002/nbm.3159, Table 2: DTI mean diffusivity of 8 thigh muscles at 3 T; mean and "
       "SD of the 8 muscle means")
_FU = ("Funck et al. 2018, doi:10.1002/mrm.26919: whole blood at 37 C, flow-compensated, suggested D_b = 1.54 +- "
       "0.12 um^2/ms (falls with haematocrit)")
_WE = ("Weidlich et al., ISMRM 2018 #2511: gluteal subcutaneous fat at 3 T, lipid ADC 1.83e-5 mm^2/s at 2 mm "
       "slices, rising with slice thickness (motion): an upper bound")
_CH = ("Chen et al. 2010, doi:10.1002/mrm.22208: water exchanges magnetization with protein and membrane "
       "protons but not with triglyceride (fat) protons")
_SM = "Smith et al., ISMRM 2005 #417: CSF MTR 0.01 +- 0.02 at 3 T (frontal WM 0.48 +- 0.01)"


def _mt_and_diffusion(strength):
    """MT and diffusion fields per tissue class; MT measured at 3 T is flagged at other fields
    (Stanisz et al. 2005 find qMT parameters at 1.5 T and 3 T similar). Diffusivity does not
    depend on B0. Classes without a semi-solid pool get bound_fraction 0 only."""
    def mt(value, sd, citation, assumed=False):
        if strength != 3.:
            citation, assumed = "Assumption: " + citation + f"; 3 T value used at {strength} T", True
        return Distribution(value, sd, citation, assumed)
    rate = mt(19., None, _YA12 + ": R = 19.0 /s")
    t2b = mt(9.7e-6, None, _YA12 + ": T2B = 9.7 us")
    out = {"wm": dict(bound_fraction=mt(.1277, .0041, _YA16 + ": WM 12.77 +- 0.41 %"), exchange_rate=rate,
                      bound_t2=t2b, diffusivity=Distribution(.70e-9, .03e-9, _HE + ": white matter")),
           "gm": dict(bound_fraction=mt(.0673, .0015, _YA16 + ": GM 6.73 +- 0.15 %"), exchange_rate=rate,
                      bound_t2=t2b, diffusivity=Distribution(.89e-9, .04e-9, _HE + ": cortical grey matter")),
           "csf": dict(bound_fraction=mt(0., None, _SM + ": no semi-solid pool"),
                       diffusivity=Distribution(3.04e-9, None, _HO)),
           "muscle": dict(bound_fraction=mt(.084, .006, _SI), bound_t2=mt(5.9e-6, .2e-6, _SI),
                          exchange_rate=_assumed(19., "no converged value in human muscle (Sinclair et al. 2010, "
                                                 "doi:10.1002/mrm.22562); the brain's R from " + _YA12),
                          diffusivity=Distribution(1.78e-9, .08e-9, _LI)),
           "fat": dict(bound_fraction=Distribution(0., None, _CH), diffusivity=Distribution(1.83e-11, None, _WE)),
           "vessels": dict(bound_fraction=_assumed(0., "blood's MT is small next to tissue's, the premise of "
                           "MT-suppressed TOF angiography (Pike et al. 1992, doi:10.1002/mrm.1910250217)"),
                           diffusivity=Distribution(1.54e-9, .12e-9, _FU))}
    for name in ("skin", "connective", "dura"):
        out[name] = {key: _assumed(d.mean, d.citation.removeprefix("Assumption: ") + f"; muscle used for {name}")
                     for key, d in out["muscle"].items()}
    out["marrow"] = {key: _assumed(d.mean, d.citation + "; fat used for marrow, a lipid class here")
                     for key, d in out["fat"].items()}
    out["dermis"] = {key: _assumed(d.mean, d.citation.removeprefix("Assumption: ") + "; muscle used for the dermis "
                                   "(collagen-rich: likely more MT, no in-vivo value found)") for key, d in out["muscle"].items()}
    out["lesion"] = {key: _assumed(d.mean * (.6 if key == "bound_fraction" else 1.),
                                   d.citation.removeprefix("Assumption: ") + ("; MS lesions: 0.6 x white matter's bound pool assumed "
                                   "(F reduced against NAWM in every qMT study, York et al. 2022 meta-analysis doi:10.1093/braincomms/fcac134; "
                                   "MPF 7.2 % in lesions against ~12 % in white matter, Sci Rep 2024 doi:10.1038/s41598-024-67445-4)"
                                   if key == "bound_fraction" else "; white matter's value used for lesions"))
                     for key, d in out["wm"].items() if key != "diffusivity"}
    out["lesion"]["diffusivity"] = _assumed(1.2e-9, "MS plaque mean diffusivity 1.23 +- 0.12 and 1.24 +- 0.11 x 1e-3 mm2/s (Chin J "
                                            "Contemp Neurol Neurosurg 2012, 32 patients; Egypt J Radiol Nucl Med 2017, 3 T)")
    out["eye"] = dict(bound_fraction=_assumed(0., "vitreous: dilute macromolecules, no semi-solid pool"),
                      diffusivity=_assumed(3.04e-9, _HO + "; free water used for the vitreous"))
    return out


DEFAULT_TISSUES, BRAINWEB_TISSUES = _tables()
