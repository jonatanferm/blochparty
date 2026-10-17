"""Bloch Party: a differentiable MRI simulator in PyTorch."""

from . import design, epg, exchange, reference
from .anatomy import (Anatomy, BRAINWEB_TISSUES, DEFAULT_TISSUES, DigitalPhantom, Distribution, TissueProperties,
                      anatomy_field_maps, deform, draw_tissues, random_deformation, sample_head, layer_scalp, add_dermis, label_deep_grey, add_falx, open_sulci, sulcal_width, sulcal_widening, gather_cortical_csf, head_dimensions, anthropometric_scaling, gaussian_random_field, load_ahead, estimate_macroscopic_r2prime, load_brainweb, load_ixi, ixi_subjects, load_qsm2,
                      to_phantom)
from .bloch import SimResult
from .sequences.library import (bssfp_2d, cartesian_kspace, cartesian_recon, epi_2d, gre_2d, gre_3d, gre_cartesian,
                       readout_bandwidth_per_pixel, se_2d, tse_2d)
from .chunked import simulate_chunked
from .engines import simulate  # engine= picks the core (default), the chunked core or a GPU engine
from .coils import add_noise, birdcage_b1, birdcage_sensitivities
from .constants import GAMMA_BAR
from .exchange import BoundPool, ExchangePool
from .maps import FieldMaps
from .motion import Motion
from .restriction import Restriction
from .phantom import (Phantom, b0_cylinder, b0_shim, b0_sphere, grid_positions, phantom_from_maps, shepp_logan,
                      shepp_logan_maps)
from .recon import coil_combine, ifft2c, ifft3c, ifftnc, kspace_grid, reconstruct, rss
from .sequence import Sequence
from .sequences.builder import SequenceBuilder, System, block_pulse, gradient_waveform, sinc_pulse, trapezoid
from .susceptibility import shim, susceptibility_to_b0

__all__ = [
    "Anatomy", "BRAINWEB_TISSUES", "DEFAULT_TISSUES", "DigitalPhantom", "Distribution", "TissueProperties",
    "anatomy_field_maps", "deform", "draw_tissues", "random_deformation", "sample_head", "layer_scalp", "add_dermis", "label_deep_grey", "add_falx", "open_sulci", "sulcal_width", "sulcal_widening", "gather_cortical_csf", "head_dimensions", "anthropometric_scaling", "gaussian_random_field", "load_ahead", "estimate_macroscopic_r2prime", "load_brainweb", "load_ixi", "ixi_subjects", "load_qsm2", "to_phantom",
    "GAMMA_BAR", "BoundPool", "ExchangePool", "FieldMaps", "Motion", "Phantom", "Restriction", "Sequence", "SequenceBuilder", "SimResult", "System", "add_noise", "b0_cylinder",
    "b0_shim", "b0_sphere", "birdcage_b1", "birdcage_sensitivities", "block_pulse", "bssfp_2d",
    "cartesian_kspace", "cartesian_recon", "coil_combine", "design", "epg", "epi_2d", "exchange",
    "gradient_waveform", "gre_2d", "gre_3d", "gre_cartesian", "grid_positions", "ifft2c", "ifft3c", "ifftnc",
    "kspace_grid", "phantom_from_maps", "readout_bandwidth_per_pixel", "reconstruct", "reference", "rss",
    "se_2d", "shepp_logan", "shepp_logan_maps", "shim", "simulate", "simulate_chunked", "sinc_pulse",
    "susceptibility_to_b0", "trapezoid",
    "tse_2d",
]
