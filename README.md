# Supplemental code and data for paper "Mechanics of exploration in Drosophila melanogaster" by Jane Loveless, Konstantinos Lagogiannis, and Barbara Webb. 

All code was written by Jane Loveless. All data files were generated by the code included in this repo, with the exception of "./data/szigeti_modes/szigeti_modes.mat" which was originally generated by Balázs Szigeti and published in "Searching for motifs in the behaviour of larval Drosophila melanogaster and Caenorhabditis elegans reveals continuity between behavioural states" by Balázs Szigeti, Ajinkya Deogade, and Barbara Webb, Journal of the Royal Society Interface.


The provided code consists of a Python module, neuromech, along with several
Python scripts which use this module to generate data, run analyses, and produce
the plots shown in the main paper.

There are a few dependencies for this code to work :
	numpy (ideally compiled with support for LAPACK/BLAS)
	scipy (ideally compiled with support for LAPACK/BLAS)
	sympy
	matplotlib (animation backend is required to reproduce supplemental
		    video content)
	odespy (available at https://github.com/hplgit/odespy; this must be
		compiled with support for the ODEPACK solvers and FORTRAN
		compilation)
	powerlaw (available at https://pypi.python.org/pypi/powerlaw)


The neuromech module is split into a set of submodules, as follows :

<b>symbol</b>
defines several useful mathematical symbols used in other submodules, e.g. t for
time, V for membrane voltage/neuron activation

<b>model</b>
defines generic model classes, mechanical models, and neural models. Also
provides some helper functions for doing classical mechanics in a Hamiltonian
framework.

<b>analysis</b>
routines for analysing models, either directly or via data generated through
simulation

<b>util</b>
utility functions and wrappers, e.g. tools for pretty plotting; tools for
numerical integration using odespy; tools for generating FORTRAN source code
compatible with odespy from a system of symbolic differential equations using
sympy

Typical workflow within a script will consist of defining symbolic mechanical
and neural models before combining them and converting the composite system of
symbolic dynamical equations into compilable FORTRAN source code for numerical
investigation. Analysis is then either done on the symbolic models directly or
the numerical models directly or on simulation outputs, or some combination
thereof.

The included scripts are  :

<b>1_modal_analysis.py</b>
corresponds to figure 3 in paper. Symbolic analysis of the body's
small-amplitude motion.

<b>2_peristalsis.py</b>
corresponds to figure 4 in paper. Numerically integrate dissipative axial
motion; bifurcation analysis; visualisation using modal coordinates

<b>3_conservative_head.py</b>
corresponds to figure 5 and 6 in paper. Analysis of head's large-amplitude
behaviour and transition to chaotic motion; uses Poincare section, LCE
estimation, power spectrum, autocorrelation

<b>4_conservative_body.py</b>
corresponds to figure 7 in paper. Analysis of full body's large-amplitude
behaviour; uses LCE estimation, power spectrum, autocorrelation

<b>5_exploration.py</b>
corresponds to figure 8 and 9 in paper. Generate trajectories for full model;
LCE estimation. This script does not perform any plotting.

<b>6_trajectory_analysis.py</b>
corresponds to figure 8 and 9 in paper. Analysis of trajectories generated by
previous script; uses power spectrum, autocorrelation, correlation dimension,
box-counting dimension, 2-segment analysis, run-length distribution, body bend
distribution, mean-squared displacement, etc.

<b>LCE_test_lorenz.py</b>
tests LCE estimation algorithm on Lorenz system

<b>LCE_test_rossler.py</b>
tests LCE estimation algorithm on Rossler system
