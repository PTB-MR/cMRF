"""Gold standard SE-based inversion recovery sequence for T1 mapping."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pypulseq as pp

from utils.preparation_blocks import add_t1prep

######################################
### BEGIN of configuration section ###
######################################

# choose flags
FLAG_PLOT_TRAJ = True  # toggle plotting of k-space trajectory
FLAG_PLOT_SEQ_DIAGRAM = True  # toggle plotting of the complete sequence diagram
FLAG_TIMINGCHECK = True  # toggle timing check of the sequence
FLAG_TESTREPORT = True  # toggle advanced test report including timing check

# define system limits and create PyPulseq sequence object
system = pp.Opts(
    max_grad=30,
    grad_unit="mT/m",
    max_slew=100,
    slew_unit="T/m/s",
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)
seq = pp.Sequence(system=system)

# choose inversion times [s]
inversion_times = np.array([25, 50, 300, 600, 1200, 2400, 4800]) * 1e-3

# define geometry parameters
fov = 128e-3  # [m]
n_x = 128  # number of points per readout
n_y = 128  # number of phase encoding points
slice_thickness = 8e-3  # [m]

# define acquisition parameters
te = 20e-3  # echo time (TE) [s]
tr = 8  # repetition time (TR) [s]

# define gradient/ADC timing
readout_gradient_prewinder_duration = 1e-3
readout_gradient_flat_time = 3.2e-3
spoiler_gradient_duration = 3.2e-3

# define T1prep settings
inversion_pulse_duration = 10.24e-3  # [s]
inversion_spoiler_rise_time = 700e-6  # [s]
inversion_spoiler_duration = 9.6e-3  # [s]

####################################
### END of configuration section ###
####################################

# create phase encoding steps
enc_steps_pe = np.arange(0, n_y)

# create slice selective excitation pulse
rf90, gz90, _ = pp.make_sinc_pulse(  # type: ignore
    flip_angle=np.pi / 2,
    system=system,
    duration=4e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    return_gz=True,
    use="excitation",
)

# manually create rephasing gradient for slice selection
gz90_reph = pp.make_trapezoid(
    channel="z",
    system=system,
    area=-gz90.area / 2,
    duration=readout_gradient_prewinder_duration,
)

# create refocusing pulse
rf180, gz180, _ = pp.make_sinc_pulse(  # type: ignore
    flip_angle=np.pi,
    system=system,
    duration=2.5e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    phase_offset=np.pi / 2,
    return_gz=True,
    use="refocusing",
)

# create readout gradient
delta_k = 1 / fov
gx = pp.make_trapezoid(
    channel="x",
    flat_area=n_x * delta_k,
    flat_time=readout_gradient_flat_time,
    system=system,
)

# create ADC event
adc = pp.make_adc(
    num_samples=n_x,
    duration=gx.flat_time,
    delay=gx.rise_time,
    system=system,
)

# calculate frequency encoding pre-winder gradient
gx_pre = pp.make_trapezoid(
    channel="x",
    area=-gx.area / 2 - delta_k / 2,
    duration=readout_gradient_prewinder_duration,
    system=system,
)

# calculate frequency encoding rewinder gradient
gx_post = pp.make_trapezoid(
    channel="x",
    area=-gx.area / 2 + delta_k / 2,
    duration=readout_gradient_prewinder_duration,
    system=system,
)

# calculate gradient areas for phase encoding direction
phase_areas = (np.arange(n_y) - n_y / 2) * delta_k

# spoiler along slice direction before and after 180°-SE-refocusing pulse
gz_spoil = pp.make_trapezoid(
    channel="z",
    system=system,
    area=gz90.area * 4,
    duration=spoiler_gradient_duration,
)

# define raster_time variable
raster_time = system.grad_raster_time


# adiabatic inversion pulse
rf_inv = pp.make_adiabatic_pulse(
    pulse_type="hypsec",
    adiabaticity=6,
    beta=800,
    mu=4.9,
    duration=inversion_pulse_duration,
    system=system,
    use="inversion",
)

gz_inv_spoil = pp.make_trapezoid(
    channel="z",
    amplitude=0.5 * system.max_grad,
    duration=inversion_spoiler_duration,
    rise_time=inversion_spoiler_rise_time,
)

# calculate delay between gz90_reph and gz_spoil
delay_gz90_reph_and_gz_spoil = (
    te / 2
    - pp.calc_duration(gz90) / 2
    - pp.calc_duration(gz90_reph)
    - pp.calc_duration(gz_spoil)
    - pp.calc_duration(gz180) / 2
)
delay_gz90_reph_and_gz_spoil = np.ceil(delay_gz90_reph_and_gz_spoil / raster_time) * raster_time

# calculate delay between gz_spoil and gx_pre
delay_gz_spoil_and_gx_pre = (
    te / 2
    - pp.calc_duration(gz180) / 2
    - pp.calc_duration(gz_spoil)
    - pp.calc_duration(gx_pre)
    - pp.calc_duration(gx) / 2
)
delay_gz_spoil_and_gx_pre = np.ceil(delay_gz_spoil_and_gx_pre / raster_time) * raster_time

# calculate delays after gz_spoil and before next TR for all inversion times
delay_tr = (
    tr
    - te
    - pp.calc_duration(gz90) / 2
    - pp.calc_duration(gx) / 2
    - pp.calc_duration(gx_post, gz_spoil)
    - inversion_times
    - pp.calc_duration(rf_inv) / 2
)
delay_tr = np.ceil(delay_tr / raster_time) * raster_time

# calculate inversion delays for the different inversion times
inversion_time_delays = inversion_times - pp.calc_duration(rf90) / 2 - pp.calc_duration(rf_inv) / 2
inversion_time_delays = np.ceil(inversion_time_delays / raster_time) * raster_time

for delay_idx, delay in enumerate(inversion_time_delays):
    contrast_label = pp.make_label(type="SET", label="ECO", value=int(delay_idx))

    for pe in enc_steps_pe:
        pe_label = pp.make_label(type="SET", label="LIN", value=int(pe))

        seq.add_block(rf_inv)
        seq.add_block(gz_inv_spoil, pp.make_delay(delay))

        # add 90° excitation rf pulse
        seq.add_block(rf90, gz90)

        # add gradients and refocusing pulse
        seq.add_block(gz90_reph)
        seq.add_block(pp.make_delay(delay_gz90_reph_and_gz_spoil))
        seq.add_block(gz_spoil)
        seq.add_block(rf180, gz180)
        seq.add_block(gz_spoil)
        seq.add_block(pp.make_delay(delay_gz_spoil_and_gx_pre))

        # calculate phase encoding gradient
        gy_pre = pp.make_trapezoid(
            channel="y",
            area=phase_areas[pe],
            duration=pp.calc_duration(gx_pre),
            system=system,
        )

        # add gx pre-winder and gy phase-encoding gradients
        seq.add_block(gx_pre, gy_pre, pe_label, contrast_label)

        # add readout gradient and ADC
        seq.add_block(gx, adc)

        # add re-winder and spoiler gradients
        gy_pre.amplitude = -gy_pre.amplitude
        seq.add_block(gx_post, gy_pre, gz_spoil)

        seq.add_block(pp.make_delay(delay_tr[delay_idx]))

    if FLAG_PLOT_TRAJ and delay_idx == 0:
        k_traj_adc, k_traj, _, _, _ = seq.calculate_kspace()

# check timing of the sequence
if FLAG_TIMINGCHECK and not FLAG_TESTREPORT:
    ok, error_report = seq.check_timing()
    if ok:
        print("\nTiming check passed successfully")
    else:
        print("\nTiming check failed! Error listing follows\n")
        print(error_report)

# show advanced rest report
if FLAG_TESTREPORT:
    print("\nCreating advanced test report...")
    print(seq.test_report())

# define filename
filename = f"inv_rec_{len(inversion_times)}vals_se_{int(fov*1000)}fov_{n_x}px"

# write all required parameters in the seq-file header/definitions dictionary
seq.set_definition("Name", "t1_IR_SE")
seq.set_definition("FOV", fov)
seq.set_definition("TE", te)
seq.set_definition("TR", tr)
seq.set_definition("slice_thickness", slice_thickness)
seq.set_definition("TI", inversion_times)
seq.set_definition("sampling_scheme", "cartesian")
seq.set_definition("number_of_readouts", int(n_x))
seq.set_definition("k_space_encoding1", int(n_y))

# save seq-file
print(f"\nSaving sequence file '{filename}.seq' in 'output' folder.")
output_path = Path.cwd() / "output"
output_path.mkdir(parents=True, exist_ok=True)
seq.write(str(output_path / filename), create_signature=True)

if FLAG_PLOT_TRAJ:
    plt.plot(k_traj[0], k_traj[1], "--", color="blue")
    plt.plot(k_traj_adc[0], k_traj_adc[1], ".", color="red")
    plt.show()

if FLAG_PLOT_SEQ_DIAGRAM:
    seq.plot(time_range=(0, tr - delay_tr[0]))
