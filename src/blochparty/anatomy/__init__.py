"""Anatomies on a physical grid (BrainWeb, QSM, AHEAD and IXI heads with cited tissue properties), the head sampler and its edits, and sub-voxel spins from them.

Fuzzy anatomy on a physical grid, conservative resampling, and sub-voxel spins.

Arrays/voxel sizes: (z,y,x); positions/origin: (x,y,z) metres. Assembly runs on
CPU; transfer the resulting Phantom to an engine afterwards. Fractions describe
piecewise-constant *densities*, not the location of a boundary inside a cell.
Use a finer anatomy grid to resolve a sharp boundary independently of reconstruction.

The package: ``grid`` (the anatomy on its grid), ``tables`` (cited tissue properties), the loaders
(``brainweb``, ``qsm``, ``ahead``, ``ixi``), ``phantom`` (anatomy to spins), ``edits`` (what the head generator
changes), ``deformation`` and ``sampler`` (``sample_head``); ``tissuedb`` is the tissue-property database.
Every name is importable from ``blochparty.anatomy`` as before.
"""

from .ahead import AHEAD_UNITS, load_ahead
from .brainweb import load_brainweb
from .deformation import _bspline_weights, deform, deformation_jacobian, random_deformation
from .edits import (
                      BRAINWEB_AGE,
                      DEEP_GREY,
                      FALX_ABOVE_CALLOSUM,
                      FALX_THICKNESS,
                      GENERATED,
                      LESION_REGIONS,
                      SCALP_LAYERS,
                      SCALP_LAYERS_SD,
                      SULCAL_WIDTH_AGE,
                      VENTRICLE_AGE,
                      add_dermis,
                      add_falx,
                      add_lesions,
                      gather_cortical_csf,
                      label_deep_grey,
                      layer_scalp,
                      ndimage_dilate,
                      open_sulci,
                      sulcal_widening,
                      sulcal_width,
                      ventricle_displacement,
                      ventricle_factor,
                      ventricle_mask,
                      ventricle_offset,
)
from .grid import AIR_CHI, WATER_CHI, Anatomy, Distribution, TissueProperties
from .ixi import IXI_MIN_LNCC, ixi_subjects, load_ixi
from .phantom import (
                      AHEAD_R2STAR_ACQUISITION,
                      DEFAULT_R2STAR_ACQUISITION,
                      anatomy_field_maps,
                      estimate_macroscopic_r2prime,
                      gaussian_random_field,
                      to_phantom,
)
from .qsm import QSM_LABELS, _file_sha256, load_qsm2
from .sampler import (
                      ANSUR_HEAD,
                      BRAINWEB_SUBJECTS,
                      DEFAULT_TEXTURE,
                      DigitalPhantom,
                      _head_mask,
                      _sample_field,
                      _shell_translation,
                      anthropometric_scaling,
                      draw_tissues,
                      head_dimensions,
                      intracranial_mask,
                      sample_head,
)
from .tables import BRAINWEB_TISSUES, DEFAULT_TISSUES

__all__ = ['AHEAD_R2STAR_ACQUISITION', 'AHEAD_UNITS', 'AIR_CHI', 'ANSUR_HEAD', 'Anatomy', 'BRAINWEB_AGE',
           'BRAINWEB_SUBJECTS', 'BRAINWEB_TISSUES', 'DEEP_GREY', 'DEFAULT_R2STAR_ACQUISITION', 'DEFAULT_TEXTURE',
           'DEFAULT_TISSUES', 'DigitalPhantom', 'Distribution', 'FALX_ABOVE_CALLOSUM', 'FALX_THICKNESS', 'GENERATED',
           'IXI_MIN_LNCC', 'LESION_REGIONS', 'QSM_LABELS', 'SCALP_LAYERS', 'SCALP_LAYERS_SD', 'SULCAL_WIDTH_AGE',
           'TissueProperties', 'VENTRICLE_AGE', 'WATER_CHI', '_bspline_weights', '_file_sha256', '_head_mask',
           '_sample_field', '_shell_translation', 'add_dermis', 'add_falx', 'add_lesions', 'anatomy_field_maps',
           'anthropometric_scaling', 'deform', 'deformation_jacobian', 'draw_tissues',
           'estimate_macroscopic_r2prime', 'gather_cortical_csf', 'gaussian_random_field', 'head_dimensions',
           'intracranial_mask', 'ixi_subjects', 'label_deep_grey', 'layer_scalp', 'load_ahead', 'load_brainweb',
           'load_ixi', 'load_qsm2', 'ndimage_dilate', 'open_sulci', 'random_deformation', 'sample_head',
           'sulcal_widening', 'sulcal_width', 'to_phantom', 'ventricle_displacement', 'ventricle_factor',
           'ventricle_mask', 'ventricle_offset']
