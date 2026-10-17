"""Sequence design: hardware / safety constraints, image-quality metrics and losses, and a
machine-readable design report.

Everything here is a differentiable function of a ``Sequence``'s tensors (or of simulated
images), so it can sit in a loss next to the simulator:

* constraints: gradient amplitude, slew rate, peak B1, RF energy / SAR proxy, a PNS
  estimate and scan time, each with a smooth penalty (``penalties``) and a report entry
  that names every violation by segment range (``check_hardware``);
* metrics: tissue contrast, CNR, SNR efficiency, PSF width / sidelobes from a simulated
  point spin, and the Cramér–Rao bound of tissue-parameter estimates (``crb``);
* ``minimize``: an augmented-Lagrangian wrapper around L-BFGS for constrained designs;
* ``design_report``: timing, every constraint with its margin, and image metrics, as a
  JSON-serialisable dict meant to be read and acted on by an agent.

Units follow the rest of the package internally (RF and gradients in Hz and Hz/m); limits
and reported values use scanner units: mT/m, T/m/s, µT. Segment boundary j is the start of
segment j (j = 0..S, boundary S is the end of the sequence); the gradient is zero before
the first and after the last segment, so a sequence must ramp up and down within itself.

The package: ``hardware`` (limits, checks, penalties), ``metrics`` (image quality, PSF, AUC), ``estimation`` (Jacobian,
Fisher information, CRB), ``optimize`` (``minimize``), ``report`` (``design_report``), ``parameters`` (sequences with
learnable parameters); ``problems`` holds worked design problems (DESPOT1 flip angles, contrast per scan time, TSE
refocusing trains), each with its reference optimum.
"""

from ..constants import GAMMA_BAR
from .estimation import crb, fisher_information, jacobian
from .hardware import (
                       AXES,
                       Check,
                       Hardware,
                       b1_uT,
                       boundary_times,
                       check_hardware,
                       grad_mT_m,
                       gradient_steps,
                       hinge2,
                       penalties,
                       pns,
                       rf_energy,
                       rf_power,
                       sar_proxy,
                       slew_rate,
                       violations,
)
from .metrics import (
                       auc,
                       bootstrap,
                       cnr,
                       contrast,
                       image_noise_sigma,
                       magnitude_image,
                       point_spread_function,
                       psf_metrics,
                       snr,
                       snr_efficiency,
                       tissue_mean,
)
from .optimize import minimize
from .parameters import Design, learnable_gre
from .report import design_report

__all__ = ['AXES', 'Check', 'GAMMA_BAR', 'Hardware', 'auc', 'b1_uT', 'bootstrap', 'boundary_times', 'check_hardware',
           'cnr', 'contrast', 'crb', 'design_report', 'fisher_information', 'grad_mT_m', 'gradient_steps', 'hinge2',
           'image_noise_sigma', 'jacobian', 'magnitude_image', 'minimize', 'penalties', 'pns',
           'point_spread_function', 'psf_metrics', 'rf_energy', 'rf_power', 'sar_proxy', 'slew_rate', 'snr',
           'snr_efficiency', 'tissue_mean', 'violations', 'Design', 'learnable_gre']
