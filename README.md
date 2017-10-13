# Supplemental code and data for paper "Mechanics of exploration in Drosophila
# melanogaster" by Jane Loveless, Konstantinos Lagogiannis, and Barbara Webb
#
#
# all code was written by Jane Loveless
#
# all data was generated by Jane Loveless, with the exception of the file
# "./data/szigeti_modes/szigeti_modes.mat". This data was originally published
# in [1].
#
# [1] "Searching for motifs in the behaviour of larval Drosophila melanogaster
# and Caenorhabditis elegans reveals continuity between behavioural states" by
# Balázs Szigeti, Ajinkya Deogade, and Barbara Webb

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

	neuromech
		symbol		: defines several useful mathematical symbols
				  used in other submodules, e.g. t for time, V
				  for membrane voltage/neuron activation
		model		: defines generic model classes
		mechanical	: defines mechanical models and provides
				  helpful functions for doing classical
				  mechanics, e.g. for generating symbolic
				  equations of motion; most of the classical
				  mechanics is coded by hand to follow a
				  Hamiltonian framework rather than relying on
				  sympy.physics.mechanics, which uses primarily
				  Newtonian or Lagrangian frameworks
		neuron		: defines neural models
		synapse		: defines synapse models and provides functions
				  for connecting neural populations
		system		: provides a framework for combining neural and
				  mechanical models
		analysis	: routines for analysing models, either directly
				  or via data generated through simulation
		util		: utility functions and wrappers, e.g. tools for
				  pretty plotting, tools for generating FORTRAN
				  source code compatible with odespy from a
				  system of symbolic differential equations
				  using sympy; tools for numerical integration
				  using odespy

Typical workflow within a script will consist of defining symbolic mechanical
and neural models (neuromech.mechanical and neuromech.neural, some use of
SymPy directly) before combining them (neuromech.synapse and neuromech.system,
some use of SymPy directly), and converting the composite system of symbolic
dynamical equations into compilable FORTRAN source code for numerical
investigation (neuromech.util). Analysis is either done on the symbolic models
directly (e.g. using sympy or some neuromech functions) or the numerical models
directly (e.g. estimation of maximal Lyapunov characteristic exponent using
neuromech.analysis.lce_estimate) or on simulation outputs (most functions in
neuromech.analysis).

The provided scripts are as follows :

	1_modal_analysis.py	 : analysis of the body's small-amplitude motion
	2_peristalsis.py	 : numerically integrate dissipative axial
				   motion; bifurcation analysis;
				   visualisation using modal coordinates
	3_conservative_head.py	 : analysis of head's large-amplitude behaviour
				   and transition to chaotic motion; uses
				   Poincare section, LCE estimation, power
				   spectrum, autocorrelation
	4_conservative_body.py	 : analysis of full body's large-amplitude
				   behaviour; uses LCE estimation, power
				   spectrum, autocorrelation
	5_exploration.py	 : generate trajectories for full model; LCE
				   estimation
	6_trajectory_analysis.py : analysis of trajectories generated by previous
				   script, except LCE estimation; uses power
				   spectrum, autocorrelation, correlation
				   dimension, box-counting dimension, 2-segment
				   analysis, run-length distribution, body bend
				   distribution, mean-squared displacement, etc.
	LCE_test_lorenz.py	 : tests LCE estimation algorithm on Lorenz
				   system
	LCE_test_rossler.py	 : tests LCE estimation algorithm on Rossler
				   system