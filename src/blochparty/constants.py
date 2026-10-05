"""Physical constants shared across modules."""

GAMMA_BAR = 42.577478518e6  # 1H gyromagnetic ratio / 2 pi [Hz/T] (CODATA)

# pypulseq's rounded value. Used only to convert hardware limits given in mT/m and T/m/s,
# so that gradient designs and System defaults match pypulseq's exactly.
PULSEQ_GAMMA_BAR = 42.576e6  # [Hz/T]
