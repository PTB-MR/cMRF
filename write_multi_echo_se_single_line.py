"""Gold standard SE-based multi-echo sequence for T2 mapping."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pypulseq as pp

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

# choose echo times [s]
echo_times = np.array([0.024, 0.05, 0.1, 0.2, 0.4])

# define geometry parameters
fov = 128e-3  # [m]
n_x = 128  # number of points per readout
n_y = 128  # number of phase encoding points
slice_thickness = 8e-3  # [m]

# define repetition time (TR)
tr = 8  # repetition time [s]

# define gradient/ADC timing
readout_gradient_prewinder_duration = 1e-3
readout_gradient_flat_time = 3.2e-3
spoiler_gradient_duration = 3.2e-3

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

# loop over all echo times
for te_idx, te in enumerate(echo_times):
    contrast_label = pp.make_label(type="SET", label="ECO", value=int(te_idx))

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

    # calculate delay after gz_spoil and before next TR
    delay_tr = tr - te - pp.calc_duration(gz90) / 2 - pp.calc_duration(gx) / 2 - pp.calc_duration(gx_post, gz_spoil)
    delay_tr = np.ceil(delay_tr / raster_time) * raster_time

    if FLAG_PLOT_SEQ_DIAGRAM and te_idx == 0:
        upper_plot_lim = tr - delay_tr

    for pe in enc_steps_pe:
        pe_label = pp.make_label(type="SET", label="LIN", value=int(pe))

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

        seq.add_block(pp.make_delay(delay_tr))

    if FLAG_PLOT_TRAJ and te_idx == 0:
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
filename = f"multi_echo_{len(echo_times)}vals_se_{int(fov*1000)}fov_{n_x}px"

# write all required parameters in the seq-file header/definitions dictionary
seq.set_definition("Name", "t2_ME_SE")
seq.set_definition("FOV", fov)
seq.set_definition("TE", echo_times)
seq.set_definition("TR", tr)
seq.set_definition("slice_thickness", slice_thickness)
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
    seq.plot(time_range=(0, upper_plot_lim))
