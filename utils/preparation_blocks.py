"""Adiabatic T1prep / inversion block & non-adiabatic composite T2prep block."""

import numpy as np
import pypulseq as pp


def _add_composite_refocusing_block(
    system: pp.Opts,
    duration_180: float,
    rf_gap_time: float,
    seq: pp.Sequence | None = None,
    negative_amp: bool = False,
) -> tuple[pp.Sequence, float]:
    """Add a 90°x, +/-180°y, 90°x refocusing block to a sequence.

    Parameters
    ----------
    system
        system limits
    duration_180
        duration of 180° refocussing block puls
    rf_gap_time
        time between 2 consecutive RF pulses
    seq
        PyPulseq Sequence object
    negative_amp
        toggles negative amplitude for 180°y pulse

    Returns
    -------
    seq
        PyPulseq Sequence object
    duration
        duration of the composite refocusing block in seconds
    """
    if not seq:
        seq = pp.Sequence()

    flip_angles = [90, 180, 90]
    durations = [duration_180 / 2, duration_180, duration_180 / 2]
    if not negative_amp:
        phases = [0, 90, 0]
    else:
        phases = [180, 270, 180]

    for n, (fa, phase, dur) in enumerate(zip(flip_angles, phases, durations, strict=True)):
        rf = pp.make_block_pulse(
            flip_angle=fa * np.pi / 180, phase_offset=phase * np.pi / 180, duration=dur, system=system
        )
        seq.add_block(rf)
        if n < len(flip_angles) - 1:
            seq.add_block(pp.make_delay(rf_gap_time))

    total_dur = 2 * duration_180 / 2 + duration_180 + 2 * rf_gap_time

    return (seq, total_dur)


def add_t1prep(
    seq: pp.Sequence | None = None,
    inversion_time: float = 21e-3,
    rf_duration: float = 10.24e-3,
    spoil_duration: float = 9.6e-3,
    spoil_ramp_time: float = 7e-4,
    system: pp.Opts | None = None,
) -> tuple[pp.Sequence, float]:
    """Add an adiabatic T1 preparation block to a sequence.

    Parameters
    ----------
    seq
        PyPulseq Sequence object
    inversion_time
        inversion time in seconds
    rf_duration
        duration of the inversion pulse
    spoil_duration
        duration of the spoiler gradient
    spoil_ramp_time
        duration of the gradient spoiling ramp
    system
        system limits


    Returns
    -------
    seq
        PyPulseq Sequence object
    total_duration
        total duration of the T2 preparation block in seconds
    """
    if not seq:
        seq = pp.Sequence()

    if not system:
        system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=100, slew_unit="T/m/s")

    # create adiabatic hyperbolic secant inversion pulse
    rf = pp.make_adiabatic_pulse(
        pulse_type="hypsec",
        adiabaticity=6,
        beta=800,
        mu=4.9,
        duration=rf_duration,
        system=system,
    )

    # create spoiler gradient
    gz_spoil = pp.make_trapezoid(
        channel="z",
        amplitude=0.5 * system.max_grad,
        duration=spoil_duration,
        rise_time=spoil_ramp_time,
    )

    # calculate inversion time delay
    time_delay = inversion_time - pp.calc_duration(rf) / 2 - pp.calc_duration(gz_spoil)

    # round delay to gradient raster time
    time_delay = np.ceil(time_delay / system.grad_raster_time) * system.grad_raster_time

    # check if delay is valid
    if not time_delay > 0:
        raise ValueError("Inversion time too short for given RF and spoiler durations.")

    # create delay event
    delay = pp.make_delay(time_delay)

    # add add events to sequence
    seq.add_block(rf)
    seq.add_block(gz_spoil)
    seq.add_block(delay)

    # calculate total duration of T1prep block
    total_duration = pp.calc_duration(rf) + pp.calc_duration(gz_spoil) + pp.calc_duration(delay)

    return (seq, total_duration)


def add_t2prep(
    seq: pp.Sequence | None = None,
    echo_time: float = 0.1,
    duration_180: float = 1e-3,
    rf_gap_time: float = 150e-6,
    spoil_ramp_time: float = 6e-4,
    spoil_flat_time: float = 6e-3,
    system: pp.Opts | None = None,
) -> tuple[pp.Sequence, float]:
    """Add a T2 preparation block to a sequence.

    Parameters
    ----------
    seq
        PyPulseq Sequence object
    echo_time
        echo time in seconds
    duration_180
        duration of 180° block pulse
    rf_gap_time
        time between 2 consecutive RF pulses
    spoil_ramp_time
        duration of gradient spoiling ramp
    spoil_flat_time
        duration of gradient spoiling flat top
    system
        system limits

    Returns
    -------
    seq
        PyPulseq Sequence object
    total_duration
        total duration of the T2 preparation block in seconds
    """
    if not seq:
        seq = pp.Sequence()

    if not system:
        system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=100, slew_unit="T/m/s")

    # add 90°x excitation pulse at the beginning
    rf_90 = pp.make_block_pulse(flip_angle=np.pi / 2, duration=duration_180 / 2, system=system)
    seq.add_block(rf_90)
    total_duration = duration_180 / 2

    # add delay before 1st MLEV-4 refocusing pulse
    delay = (
        echo_time / 8 - duration_180 / 4 - duration_180 / 2 - rf_gap_time - duration_180 / 2
    )  # TE/8 - 90°x/4 - 180°x/2 - rf_gap - 180°x/2
    if delay < 0:
        raise ValueError("Echo time too short for T2 preparation block.")
    seq.add_block(pp.make_delay(delay))
    total_duration += delay

    # add first MLEV-4 refocusing pulse
    seq, refoc_dur = _add_composite_refocusing_block(
        system=system,
        duration_180=duration_180,
        rf_gap_time=rf_gap_time,
        seq=seq,
        negative_amp=False,
    )
    total_duration += refoc_dur

    # add delay before 2nd MLEV-4 refocusing pulse
    delay = echo_time / 4 - refoc_dur
    if delay < 0:
        raise ValueError("Echo time too short for T2 preparation block.")
    seq.add_block(pp.make_delay(delay))
    total_duration += delay

    # add second MLEV-4 refocusing pulse
    seq, refoc_dur = _add_composite_refocusing_block(
        system=system,
        duration_180=duration_180,
        rf_gap_time=rf_gap_time,
        seq=seq,
        negative_amp=False,
    )
    total_duration += refoc_dur

    # add delay before 3rd MLEV-4 refocusing pulse
    delay = echo_time / 4 - refoc_dur
    seq.add_block(pp.make_delay(delay))
    total_duration += delay

    # add third MLEV-4 refocusing pulse
    seq, refoc_dur = _add_composite_refocusing_block(
        system=system,
        duration_180=duration_180,
        rf_gap_time=rf_gap_time,
        seq=seq,
        negative_amp=True,
    )
    total_duration += refoc_dur

    # add delay before 4th MLEV-4 refocusing pulse
    delay = echo_time / 4 - refoc_dur
    seq.add_block(pp.make_delay(delay))
    total_duration += delay

    # add fourth MLEV-4 refocusing pulse
    seq, refoc_dur = _add_composite_refocusing_block(
        system=system,
        duration_180=duration_180,
        rf_gap_time=rf_gap_time,
        seq=seq,
        negative_amp=True,
    )
    total_duration += refoc_dur

    # add delay before first tip-up pulse
    delay = echo_time / 8 - refoc_dur / 2 - duration_180 / 2 * 3 / 2  # TE/8 - refoc_dur/2 - 270°x/2
    if delay < 0:
        raise ValueError("Echo time too short for T2 preparation block.")
    seq.add_block(pp.make_delay(delay))
    total_duration += delay

    # add composite tip-up pulse (270°x + [-360]°x)
    rf_tip_up_270 = pp.make_block_pulse(flip_angle=3 * np.pi / 2, duration=duration_180 / 2 * 3, system=system)
    rf_tip_up_360 = pp.make_block_pulse(flip_angle=-2 * np.pi, duration=duration_180 * 2, system=system)
    seq.add_block(rf_tip_up_270)
    seq.add_block(pp.make_delay(rf_gap_time))
    seq.add_block(rf_tip_up_360)
    total_duration += duration_180 / 2 * 3 + rf_gap_time + duration_180 * 2

    # add spoiler gradient
    gz_spoil = pp.make_trapezoid(
        channel="z",
        amplitude=0.5 * system.max_grad,
        flat_time=spoil_flat_time,
        rise_time=spoil_ramp_time,
        fall_time=spoil_ramp_time,
        system=system,
    )
    seq.add_block(gz_spoil)
    total_duration += pp.calc_duration(gz_spoil)

    return (seq, total_duration)
