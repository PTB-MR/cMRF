"""Cardiac MR Fingerprinting (cMRF) sequence with variable density spiral (VDS) readout."""

from pathlib import Path

import ismrmrd
import matplotlib.pyplot as plt
import numpy as np
import pypulseq as pp
from utils.create_ismrmrd_header import create_hdr
from utils.preparation_blocks import add_t1prep, add_t2prep
from utils.vds import vds

######################################
### BEGIN of configuration section ###
######################################

# choose flags
FLAG_PLOT_SINGLE_TRAJ = True  # toggle plotting of the k-space trajectory of the first repetition
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

# define inversion time for T1prep
inversion_time = 21e-3

# define echo times for T2prep
echo_times = [0.03, 0.05, 0.1]

# cMRF specific settings
n_blocks = 15  # number of heartbeat blocks
n_unique_spirals = 48  # number of unique spiral interleaves

# define ECG trigger delay
trig_delay = 400e-3  # delay between detected trigger event and the start of the next block

# define geometry parameters
fov = 300e-3  # field of view [m]
n_x = 192  # desired image matrix size
res = fov / n_x  # spatial resolution [m]
slice_thickness = 8e-3  # slice thickness [m]

# define repetition time (TR)
tr = 10e-3  # repetition time [s]. Set to None for minimum TR

# define VDS readout parameters
if fov == 128e-3 and n_x == 128:
    n_spirals_for_traj_calc = 16
elif fov == 300e-3 and n_x == 192:
    n_spirals_for_traj_calc = 24
else:
    print("Please double check the number of spirals for trajectory calculation.")
fov_coefficients = [fov, -3 / 4 * fov]  # FOV decreases linearly from fov_coeff[0] to fov_coeff[0]-fov_coeff[1].

# define rf pulse parameters
rf_dur = 0.8e-3  # duration of excitation pulse [s]
rf_bwt_prod = 8  # bandwidth time product of rf pulses.
rf_apodization = 0.5  # rf apodization factor
rf_spoiling_inc = 0  # rf spoiling phase increment. Choose 0 to disable rf spoiling. [°]

# define set label time
time_label = 1e-5  # time for setting labels [s]

####################################
### END of configuration section ###
####################################

# import cMRF flip angle pattern from txt file
flip_angle_all = np.loadtxt(Path(__file__).parent / "utils" / "flip_angle_pattern.txt")

# read maximum flip angle from flip angle pattern
rf_phi_max = np.max(flip_angle_all)

# make sure the number of blocks fits the total number of flip angles / repetitions
if not flip_angle_all.size % n_blocks == 0:
    raise ValueError("Number of repetitions must be a multiple of the number of blocks.")

# calculate number of shots / repetitions per block
n_shots_per_block = flip_angle_all.size // n_blocks

# create rf dummy pulse (required for some timing calculations)
rf_dummy, gz_dummy, gzr_dummy = pp.make_sinc_pulse(  # type: ignore
    flip_angle=90 / 180 * np.pi,
    duration=rf_dur,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=rf_bwt_prod,
    system=system,
    return_gz=True,
    use="excitation",
)

# calculate variable density spiral (VDS) trajectory
r_max = 0.5 / fov * n_x  # [1/m]
k, g, s, timing, r, theta = vds(
    smax=system.max_slew * 0.9,
    gmax=system.max_grad * 0.9,
    T=system.grad_raster_time,
    N=n_spirals_for_traj_calc,
    Fcoeff=fov_coefficients,
    rmax=r_max,
    oversampling=12,
)

# calculate angular increment
delta_unique_spirals = 2 * np.pi / n_unique_spirals
delta_array = np.arange(0, 2 * np.pi, delta_unique_spirals)

# calculate ADC
adc_dwell = system.grad_raster_time
adc_total_samples = np.shape(g)[0] - 1
assert adc_total_samples <= 8192, "ADC samples exceed maximum value of 8192."
adc = pp.make_adc(num_samples=adc_total_samples, dwell=adc_dwell, system=system)

# Pre-calculate the n_unique_spirals gradient waveforms, k-space trajectories, and rewinders
n_points_g = np.shape(g)[0]
n_points_k = np.shape(k)[0]

spiral_readout_grad = np.zeros((n_unique_spirals, 2, n_points_g))
spiral_trajectory = np.zeros((n_unique_spirals, 2, n_points_k))
gx_readout_list = []
gy_readout_list = []
gx_rewinder_list = []
gy_rewinder_list = []
max_rewinder_duration = 0

# iterate over all unique spirals
for n, delta in enumerate(delta_array):
    exp_delta = np.exp(1j * delta)
    exp_delta_pi = np.exp(1j * (delta + np.pi))

    spiral_readout_grad[n, 0, :] = np.real(g * exp_delta)
    spiral_readout_grad[n, 1, :] = np.imag(g * exp_delta)
    spiral_trajectory[n, 0, :] = np.real(k * exp_delta_pi)
    spiral_trajectory[n, 1, :] = np.imag(k * exp_delta_pi)

    gx_readout = pp.make_arbitrary_grad(
        channel="x",
        waveform=spiral_readout_grad[n, 0],
        first=0,
        delay=adc.delay,
        system=system,
    )

    gy_readout = pp.make_arbitrary_grad(
        channel="y",
        waveform=spiral_readout_grad[n, 1],
        first=0,
        delay=adc.delay,
        system=system,
    )

    gx_rewinder, _, _ = pp.make_extended_trapezoid_area(
        area=-gx_readout.area,
        channel="x",
        grad_start=gx_readout.last,
        grad_end=0,
        system=system,
    )

    gy_rewinder, _, _ = pp.make_extended_trapezoid_area(
        area=-gy_readout.area,
        channel="y",
        grad_start=gy_readout.last,
        grad_end=0,
        system=system,
    )

    gx_readout_list.append(gx_readout)
    gy_readout_list.append(gy_readout)
    gx_rewinder_list.append(gx_rewinder)
    gy_rewinder_list.append(gy_rewinder)

    # update maximum rewinder duration
    max_rewinder_duration = max(max_rewinder_duration, pp.calc_duration(gx_rewinder, gy_rewinder))

# gradient spoiling
gz_spoil_area = 4 / slice_thickness - gz_dummy.area / 2
gz_spoil = pp.make_trapezoid(channel="z", area=gz_spoil_area, system=system)

# update maximum rewinder duration including spoiling gradient
max_rewinder_duration = max(max_rewinder_duration, pp.calc_duration(gz_spoil))

# calculate minimum echo time (TE) for sequence header
min_TE = pp.calc_duration(gz_dummy) / 2 + pp.calc_duration(gzr_dummy) + adc.delay
min_TE = np.ceil(min_TE / system.grad_raster_time) * system.grad_raster_time  # put on gradient raster

# calculate minimum repetition time (TR)
min_tr = (
    pp.calc_duration(rf_dummy, gz_dummy)  # rf pulse
    + pp.calc_duration(gzr_dummy)  # slice selection re-phasing gradient
    + pp.calc_duration(gx_readout_list[0])  # readout
    + max_rewinder_duration  # max of rewinder gradients / gz_spoil durations
    + time_label  # min time to set labels
)

# ensure minimum TR is on gradient raster
min_tr = np.ceil(min_tr / system.grad_raster_time) * system.grad_raster_time

# calculate TR delay
if tr is None:
    tr_delay = time_label
else:
    tr_delay = np.ceil((tr - min_tr + time_label) / system.grad_raster_time) * system.grad_raster_time

assert tr_delay >= time_label, f"TR must be larger than {min_tr * 1000:.2f} ms. Current value is {tr * 1000:.2f} ms."

# print TE / TR values
final_tr = min_tr if tr is None else (min_tr - time_label) + tr_delay
print("\n Manual timing calculations:")
print(f"\n shortest possible TR = {min_tr * 1000:.2f} ms")
print(f"\n final TR = {final_tr * 1000:.2f} ms")

# choose initial rf phase offset
rf_phase = 0
rf_inc = 0

# # # # # # # # # # # # #
# CREATE ISMRMRD HEADER #
# # # # # # # # # # # # #

# define full filename
filename = f"spiral_cMRF_{fov * 1000:.0f}fov_{n_x}px_{flip_angle_all.size}rep_trig{int(trig_delay * 1000)}ms"

# create folder for seq and header file
output_path = Path.cwd() / "output" / filename
output_path.mkdir(parents=True, exist_ok=True)

# delete existing header file
if (output_path / f"{filename}_header.h5").exists():
    (output_path / f"{filename}_header.h5").unlink()

# create header
hdr = create_hdr(
    traj_type="spiral",
    fov=fov,
    res=res,
    slice_thickness=slice_thickness,
    dt=adc_dwell,
    n_k1=flip_angle_all.size,
)

# write header to file
prot = ismrmrd.Dataset(output_path / f"{filename}_header.h5", "w")
prot.write_xml_header(hdr.toXML("utf-8"))

# # # # # # # # # # # # # # # #
# ADD ALL BLOCKS TO SEQUENCE  #
# # # # # # # # # # # # # # # #

# initialize LIN label
seq.add_block(pp.make_delay(time_label), pp.make_label(label="LIN", type="SET", value=0))

# initialize repetition counter
rep_counter = 0

# loop over all blocks
for block in range(n_blocks):
    # add inversion pulse for every fifth block
    if block % 5 == 0:
        # get prep block duration and calculate corresponding trigger delay
        t1prep_block, prep_dur = add_t1prep(inversion_time=inversion_time, system=system)
        current_trig_delay = trig_delay - prep_dur

        # add trigger
        seq.add_block(pp.make_trigger(channel="physio1", duration=current_trig_delay))

        # add all events of T1prep block
        for idx in t1prep_block.block_events:
            seq.add_block(t1prep_block.get_block(idx))

    # add no preparation for every block following an inversion block
    elif block % 5 == 1:
        # add trigger with chosen trigger delay
        seq.add_block(pp.make_trigger(channel="physio1", duration=trig_delay))

    # add T2prep for every other block
    else:
        # get echo time for current block
        echo_time = echo_times[block % 5 - 2]

        # get prep block duration and calculate corresponding trigger delay
        t2prep_block, prep_dur = add_t2prep(echo_time=echo_time, system=system)
        current_trig_delay = trig_delay - prep_dur

        # add trigger
        seq.add_block(pp.make_trigger(channel="physio1", duration=current_trig_delay))

        # add all events of T2prep block
        for idx in t2prep_block.block_events:
            seq.add_block(t2prep_block.get_block(idx))

    # loop over shots / repetitions per block
    for _ in range(n_shots_per_block):
        # get current flip angle
        fa = flip_angle_all[rep_counter]

        # calculate theoretical golden angle rotation for current shot
        golden_angle = (rep_counter * 2 * np.pi * (1 - 2 / (1 + np.sqrt(5)))) % (2 * np.pi)

        # find closest unique spiral to current golden angle rotation
        diff = np.abs(delta_array - golden_angle)
        idx = np.argmin(diff)

        # create slice selective rf pulse for current shot
        rf_n, gz_n, gzr_n = pp.make_sinc_pulse(  # type: ignore
            flip_angle=fa / 180 * np.pi,
            duration=rf_dur,
            slice_thickness=slice_thickness,
            apodization=rf_apodization,
            time_bw_product=rf_bwt_prod,
            system=system,
            return_gz=True,
            use="excitation",
        )

        # set current phase_offset if rf_spoiling is activated
        if rf_spoiling_inc > 0:
            rf_n.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi

        # add slice selective excitation pulse
        seq.add_block(rf_n, gz_n)

        # add slice selection re-phasing gradient
        seq.add_block(gzr_n)

        # add readout gradients and ADC
        seq.add_block(gx_readout_list[idx], gy_readout_list[idx], adc)

        # add rewinder gradients and spoiler
        gx_rewinder = gx_rewinder_list[idx]
        gy_rewinder = gy_rewinder_list[idx]
        seq.add_block(gx_rewinder, gy_rewinder, gz_spoil)

        # calculate rewinder delay for current shot
        current_rewinder_duration = max(pp.calc_duration(gx_rewinder), pp.calc_duration(gy_rewinder))
        rewinder_delay = max_rewinder_duration - current_rewinder_duration

        # add TR delay and LIN label
        seq.add_block(pp.make_delay(rewinder_delay + tr_delay), pp.make_label(label="LIN", type="INC", value=1))

        # add trajectory to ISMRMRD header
        acq = ismrmrd.Acquisition()
        acq.resize(trajectory_dimensions=2, number_of_samples=adc.num_samples)
        traj_ismrmrd = np.stack([spiral_trajectory[idx, 0, 0:-1] * fov, spiral_trajectory[idx, 1, 0:-1] * fov]).T
        acq.traj[:] = traj_ismrmrd
        prot.append_acquisition(acq)

        # update rf phase offset for the next shot
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

        if FLAG_PLOT_SINGLE_TRAJ and rep_counter == 0:
            k_traj_adc, k_traj, _, _, _ = seq.calculate_kspace()

        # increment repetition counter
        rep_counter += 1

# close ISMRMRD header file
prot.close()

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

# write all important parameters into the seq-file definitions
tr_value = tr if tr is not None else min_tr
seq.set_definition("Name", "cMRF_spiral")
seq.set_definition("FOV", [fov, fov, slice_thickness])
seq.set_definition("TE", echo_times)
seq.set_definition("TI", inversion_time)
seq.set_definition("TR", tr)
seq.set_definition("slice_thickness", slice_thickness)
seq.set_definition("sampling_scheme", "spiral")
seq.set_definition("number_of_readouts", int(n_x))

# save seq-file
print(f"\nSaving sequence file '{filename}.seq' in 'output' folder.")
seq.write(str(output_path / filename), create_signature=True)

if FLAG_PLOT_SINGLE_TRAJ:
    plt.plot(k_traj[0], k_traj[1], "--", color="blue")
    plt.plot(k_traj_adc[0], k_traj_adc[1], "o", color="red")
    plt.show()

if FLAG_PLOT_SEQ_DIAGRAM:
    seq.plot()
