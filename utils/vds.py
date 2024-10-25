"""Function to generate a variable density spiral trajectory.

Program translated from the matlab Program of Brian Hargreaves:
http://mrsrl.stanford.edu/~brian/vdspiral/

Following modifications have been made:
- Adaptation was also made from the original program to change the default unit of the matlab program
and were corrected in the description of the program underneath
-> In this new version the smax and gmax values are given in Hz/m/s and Hz/m/s and already consider the gamma factor
- Oversampling is now a parameter of the function
- Possibility to turn off/on plots and prints with a boolean

#Description given by Brian Hargreaves and where units were adapted to the modifications done
%	function [k,g,s,time,r,theta] = vds(smax,gmax,T,N,Fcoeff,rmax)
%
%	VARIABLE DENSITY SPIRAL GENERATION:
%	----------------------------------
%
%	Function generates variable density spiral which traces
%	out the trajectory
%
%			k(t) = r(t) exp(i*q(t)), 		[1]
%
%	Where q is the same as theta...
%		r and q are chosen to satisfy:
%
%		1) Maximum gradient amplitudes and slew rates.
%		2) Maximum gradient due to FOV, where FOV can
%		   vary with k-space radius r/rmax, as
%
%			FOV(r) = Sum    Fcoeff(k)*(r/rmax)^(k-1)   [2]
%
%
%	INPUTS:
%	-------
%	smax = maximum slew rate in Hz/m/s
%	gmax = maximum gradient in Hz/m (limited by Gmax or FOV)
%	T = sampling period (s) for gradient AND acquisition.
%	N = number of interleaves.
%	Fcoeff = FOV coefficients with respect to r - see above.
%	rmax= value of k-space radius at which to stop (m^-1).
%		rmax = 1/(2*resolution)
%
%
%	OUTPUTS:
%	--------
%	k = k-space trajectory (kx+iky) in m-1.
%	g = gradient waveform (Gx+iGy) in Hz/m.
%	s = derivative of g (Sx+iSy) in Hz/m/s.
%	time = time points corresponding to above (s).
%	r = k-space radius vs time (used to design spiral)
%	theta = atan2(ky,kx) = k-space angle vs time.
%
%
%	METHODS:
%	--------
%	Let r1 and r2 be the first derivatives of r in [1].
%	Let q1 and q2 be the first derivatives of theta in [1].
%	Also, r0 = r, and q0 = theta - sometimes both are used.
%	F = F(r) defined by Fcoeff.
%
%	Differentiating [1], we can get G = a(r0,r1,q0,q1,F)
%	and differentiating again, we get S = b(r0,r1,r2,q0,q1,q2,F)
%
%	(functions a() and b() are reasonably easy to obtain.)
%
%	FOV limits put a constraint between r and q:
%
%		dr/dq = N/(2*pi*F)				[3]
%
%	We can use [3] and the chain rule to give
%
%		q1 = 2*pi*F/N * r1				[4]
%
%	and
%
%		q2 = 2*pi/N*dF/dr*r1^2 + 2*pi*F/N*r2		[5]
%
%
%
%	Now using [4] and [5], we can substitute for q1 and q2
%	in functions a() and b(), giving
%
%		G = c(r0,r1,F)
%	and 	S = d(r0,r1,r2,F,dF/dr)
%
%
%	Using the fact that the spiral should be either limited
%	by amplitude (Gradient or FOV limit) or slew rate, we can
%	solve
%		|c(r0,r1,F)| = |Gmax|  				[6]
%
%	analytically for r1, or
%
%	  	|d(r0,r1,r2,F,dF/dr)| = |Smax|	 		[7]
%
%	analytically for r2.
%
%	[7] is a quadratic equation in r2.  The smaller of the
%	roots is taken, and the np.real part of the root is used to
%	avoid possible numeric errors - the roots should be np.real
%	always.
%
%	The choice of whether or not to use [6] or [7], and the
%	solving for r2 or r1 is done by findq2r2 - in this .m file.
%
%	Once the second derivative of theta(q) or r is obtained,
%	it can be integrated to give q1 and r1, and then integrated
%	again to give q and r.  The gradient waveforms follow from
%	q and r.
%
%	Brian Hargreaves -- Sept 2000.
%
%	See Brian's journal, Vol 6, P.24.
%
% ===========================================================
"""

import numpy as np


def qdf(a: float, b: float, c: float) -> tuple[float, float]:
    """Return the roots of a 2nd degree polynom ax**2+bx+c.

    Parameters
    ----------
    a : float
    b : float
    c: float

    Returns
    -------
    tuple(root1, root2): tuple(float,float)
    """
    d = b**2 - 4 * a * c
    roots = ((-b + np.sqrt(d)) / (2 * a), (-b - np.sqrt(d)) / (2 * a))

    return roots


def findq2r2(
    smax: float,
    gmax: float,
    r: float,
    r1: float,
    T: float,
    Ts: float,
    N: int,
    Fcoeff: list,
    rmax: float,
) -> tuple[float, float]:
    """Help function for vds.

    The function calculates the second derivative of the angle theta (q) and the second
    derivative of the radius r in the spiral trajectory to be integrated to
    have the angle and radius increment.

    Parameters
    ----------
    smax : float - maximal slew rate of the system in Hz/m/s
    gmax : float - maximal gradient amplitude in Hz/m
    r : float - radius of the spiral being constructed in m
    r1 : float - derivative of the radius of the spiral being constructed in m
    T : float - sampling period (s) for gradient AND acquisition.
    Ts : float - sampling period (s) for gradient AND acquisition divided by an oversampling period
    N : int - number of interleaves
    Fcoeff : list - numbers between which the FOV varies
    rmax : float - maximal radius in k-sapce  in m^(-1)

    Returns
    -------
    tuple(q2, r2): tuple(float,float) - rad/s^(-2), m/s^(-2)
    """
    F = 0  # FOV function value for this r.
    dFdr = 0  # dFOV/dr for this value of r.

    for rind in range(len(Fcoeff)):
        F += Fcoeff[rind] * (r / rmax) ** rind
        if rind > 0:
            dFdr += rind * Fcoeff[rind] * (r / rmax) ** (rind - 1) / rmax

    GmaxFOV = 1 / F / Ts
    Gmax = min(GmaxFOV, gmax)

    maxr1 = np.sqrt(Gmax**2 / (1 + (2 * np.pi * F * r / N) ** 2))
    if r1 > maxr1:
        # Grad amplitude limited.  Here we just run r upward as much as we can without
        # going over the max gradient.
        r2 = (maxr1 - r1) / T
    else:
        twopiFoN = 2 * np.pi * F / N
        twopiFoN2 = twopiFoN**2
        # A, B, C are coefficients of the equation which equates the slew rate
        # calculated from r, r1, r2 with the maximum gradient slew rate.
        # A * r2 * r2 + B * r2 + C = 0
        # A, B, C are in terms of F, dF / dr, r, r1, N and smax.
        A = r * r * twopiFoN2 + 1
        B = 2 * twopiFoN2 * r * r1 * r1 + 2 * twopiFoN2 / F * dFdr * r * r * r1 * r1
        C = (
            twopiFoN2**2 * r * r * r1**4
            + 4 * twopiFoN2 * r1**4
            + (2 * np.pi / N * dFdr) ** 2 * r * r * r1**4
            + 4 * twopiFoN2 / F * dFdr * r * r1**4
            - smax**2
        )
        rts = qdf(A, B, C)
        r2 = np.real(rts[0])
        slew = r2 - twopiFoN2 * r * r1**2 + 1j * twopiFoN * (2 * r1**2 + r * r2 + dFdr / F * r * r1**2)
        sr = np.abs(slew) / smax

        if np.abs(slew) / smax > 1.01:
            print('Slew violation, slew = ', round(abs(slew)), ' smax= ', round(smax), ' sr=', sr, ' r=', r, ' r1=', r1)

    q2 = 2 * np.pi / N * dFdr * r1**2 + 2 * np.pi * F / N * r2

    return q2, r2


def vds(
    smax: float,
    gmax: float,
    T: float,
    N: int,
    Fcoeff: list,
    rmax: float,
    oversampling: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the variable density spiral trajectory."""
    # calculate time step with oversampling
    delta_t_os = T / oversampling

    # Initialize variables
    q0 = q1 = r0 = r1 = 0

    # Initialize lists for storing the trajectory
    theta = [q0]
    r = [r0]

    while r0 < rmax:
        q2, r2 = findq2r2(smax, gmax, r0, r1, delta_t_os, T, N, Fcoeff, rmax)

        # Integrate for r, r', theta and theta'
        q1 = q1 + q2 * delta_t_os
        q0 = q0 + q1 * delta_t_os

        r1 += r2 * delta_t_os
        r0 += r1 * delta_t_os

        # Store
        theta.append(q0)
        r.append(r0)

    count = len(r)
    theta = np.array(theta)[:, np.newaxis]
    r = np.array(r)[:, np.newaxis]
    time = np.arange(count)[:, np.newaxis] * delta_t_os

    r = r[round(oversampling / 2) : count : oversampling]
    theta = theta[round(oversampling / 2) : count : oversampling]
    time = time[round(oversampling / 2) : count : oversampling]

    # Keep the length a multiple of 4, to save pain..!
    count_4 = 4 * int(np.floor(np.shape(theta)[0] / 4))
    r, theta, time = r[:count_4], theta[:count_4], time[:count_4]

    # Compute k-space trajectory on regular raster
    k = r * np.exp(1j * theta)

    # Calculate gradients by shifting k forward and backward.
    k_shifted_forward = np.vstack([np.zeros((1, 1), dtype=complex), k])
    k_shifted_backward = np.vstack([k, np.zeros((1, 1), dtype=complex)])
    g = (k_shifted_forward - k_shifted_backward)[:-1] / T

    # re-calculate trajectory values at the correct time points (1/2 * T, 3/2 * T, ...)
    # the first point is actually the ramp from 0 to g[0] over T/2. Thus the total factor 1/4.
    k = -np.cumsum(np.concatenate(([g[0] * T / 4], (g[:-1] + g[1:]) * T / 2)))

    # Compute the slew rate s
    s = -np.diff(np.vstack([np.zeros((1, 1), dtype=complex), g]), axis=0) / T

    return k.flatten(), g.flatten(), s.flatten(), time.flatten(), r.flatten(), theta.flatten()
