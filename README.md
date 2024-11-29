# Open-source cardiac Magnetic Resonance Fingerprinting (MRF)

In this repository, you can find a [(Py)Pulseq](https://github.com/imr-framework/pypulseq) implementation of a cardiac MRF sequence and the corresponding image reconstruction and dictionary matching methods.

The MRF sequence utilises inversion pulses, T2-preparation pulses and variable flip angles to increase it's sensitivity towards T1 and T2 relaxation times. Cardiac triggering is used to synchronise the acquisition with the heart beat. Data is obtained with a variable density spiral approach. Image reconstruction and dictionary matching is performed using our Python package [MRpro](https://github.com/PTB-MR/mrpro). The dictionary of the different fingerprints is calculated using an Extended Phase Graphs (EPG) approach.

In addition to the cardiac MRF framework, we also provide [(Py)Pulseq](https://github.com/imr-framework/pypulseq) implementations of commonly used spin-echo reference sequences, which can be used to validate the cardiac MRF sequence in phantom experiments.

The data needed to reproduce the phantom experiments can be found here: https://doi.org/10.5281/zenodo.14210046 
