# Open-source cardiac Magnetic Resonance Fingerprinting (MRF)

Here you can find a PyPulseq implementation of a cardiac MRF sequence and the corresponding image reconstruction and dictionary matching methods.

The MRF sequence utilises inversion pulses, T2-preparation pulses and variable flip angles to obtain data sensitive to T1 and T2. Cardiac triggering is used to synchronise the acquisition with the heart beat. Data is obtained with a variable density spiral approach. Image reconstruction and dictionary matching utilises https://github.com/PTB-MR/mrpro. The dictionary of the different fingerprints is calculated using an Extended Phase Graphs approach.

In addition to the cardiac MRF framework we also provide commonly used spin-echo reference sequences which can be used to evaluate the cardiac MRF sequence in phantom experiments.

The data needed to reproduce the phantom experiments can be found here: https://doi.org/10.5281/zenodo.14210046 
