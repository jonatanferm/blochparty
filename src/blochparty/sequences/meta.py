"""The keys of ``Sequence.meta``: what each means and who writes it. Units are SI (s, m, Hz).

``tests/sequences/test_meta.py`` checks that every key the package writes into a sequence's ``meta`` is listed here
(``Anatomy.meta`` and ``DigitalPhantom.meta`` are other dictionaries, documented in ``anatomy``).
"""

KEYS = {
    # SequenceBuilder.build
    "blocks": "the builder's blocks, in order: {'kind', 'span': (first segment, end segment), events}",
    "system": "the System (hardware limits) the sequence was built with",
    # Sequence methods
    "merged_free": "True on merge_free()'s result: RF-free runs merged into one segment each (blocks dropped)",
    # read_pulseq
    "adc_per_readout": "the ADC sample count of every block with an ADC",
    "block_segments": "(first, end) segment span of every Pulseq block",
    "segment_block": "the Pulseq block index of every segment",
    "definitions": "the file's [DEFINITIONS] section",
    "convention": "the phase convention it was read in: 'physical' or 'mrzero'",
    "pulseq_source": "the pypulseq Sequence it was read from (physical convention: write_pulseq re-emits it)",
    "pulseq_fingerprint": "a hash of the segments as read: write_pulseq rewrites an unchanged import losslessly",
    "kspace_trajectory": "(n_adc, 3) pypulseq's calculate_kspace at the ADC samples [1/m]",
    # the library (gre_cartesian, se_2d, tse_2d, bssfp_2d, epi_2d)
    "kspace_shape": "the Cartesian k-space shape in sample order, (ny, nx) or (nz, ny, nx): GRE, SE, bSSFP",
    "matrix": "(phase-encode lines, readout samples) of TSE and EPI, with kspace_index",
    "kspace_index": "the grid index of every readout line (TSE and EPI acquire lines out of order)",
    "fov": "field of view [m]",
    "te": "echo time [s] (TSE: the effective echo time of the centre line)",
    "tr": "repetition time [s]",
    "dwell": "ADC dwell time [s] (with oversampling: of the oversampled samples)",
    "esp": "echo spacing [s] (TSE, EPI)",
    "etl": "echo train length (TSE)",
    "echo_times": "the echo time of every echo in the train [s] (TSE)",
    "line_echo": "the echo index of every phase-encode line (TSE)",
    "ordering": "the TSE phase-encode ordering: 'linear', 'centric' or 'reverse_centric'",
    "slice_order": "the other slices' offsets from this one in acquisition order, this slice 0.0 (TSE multislice)",
    "slice_thickness": "slice thickness [m]; None for non-selective excitation",
    "slice_offset": "the slice's offset along z [m]",
    "ramps": "whether gradients have trapezoid ramps (False: ideal rectangular lobes)",
    "spoiling": "'ideal', 'gradient' or 'rf' (GRE)",
    "rf_phases": "the RF phase of every excitation [rad] (RF spoiling, phase cycling)",
    "phase_cycle_deg": "the bSSFP phase increment per TR [deg]",
    "readout_oversampling": "readout oversampling factor (TSE; absent when 1)",
    "bandwidth_per_pixel_pe": "bandwidth per pixel along the phase-encode direction [Hz] (EPI)",
}
