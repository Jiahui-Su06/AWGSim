import numpy as np
import types
import math
from scipy.optimize import root


def slabindex(lmbda0,t,na,nc,ns,**kwargs):
	""" Slabkwargsdex Guided mode effective index of planar waveguide.
	
	DESCRIPTION:
	Solves for the TE (or TM) effective index of a 3-layer slab waveguide
	          na          y
	  ^   ----------      |
	  t       nc          x -- z
	  v   ----------     
	          ns
	
	  with propagation in the +z direction

	INPUT:
	lambda0 - freespace wavelength
	t  - core (guiding layer) thickness
	na - cladding index (number|function)
	nc - core index (number|function)
	ns - substrate index (number|function)
	
	OPTIONS:
	Modes - max number of modes to solve
	Polarisation - one of 'TE' or 'TM'
	
	OUTPUT:
	
	neff - vector of indexes of each supported mode
	
	NOTE: it is possible to provide a function of the form n = lambda lambda0: func(lambda0) for 
	the refractive index which will be called using lambda0."""
	
	neff = []

	if "Modes" not in kwargs.keys():
		kwargs["Modes"] = np.inf

	if "Polarisation" not in kwargs.keys():
		kwargs["Polarisation"] = "TE"


	if (type(na) == types.FunctionType) or (str(type(na)) == "<class 'material.Material.Material'>"):
		na = na(lmbda0)
	if (type(nc) == types.FunctionType) or (str(type(nc)) == "<class 'material.Material.Material'>"):
		nc = nc(lmbda0)
	if (type(ns) == types.FunctionType) or (str(type(ns)) == "<class 'material.Material.Material'>"):
		ns = ns(lmbda0)

	a0 = max(np.arcsin(ns/nc),np.arcsin(na/nc))
	if np.imag(a0) != 0:
		return neff

	if kwargs["Polarisation"].upper() == "TE":
		B1 = lambda a : np.sqrt(((ns/nc)**2 - np.sin(a)**2)+0j)
		r1 = lambda a : (np.cos(a)-B1(a))/(np.cos(a)+B1(a))

		B2 = lambda a : np.sqrt(((na/nc)**2 - np.sin(a)**2)+0j)
	else:
		B1 = lambda a : (nc/ns)**2*np.sqrt(((ns/nc)**2 - np.sin(a)**2)+0j)
		r1 = lambda a : (np.cos(a)-B1(a))/(np.cos(a)+B1(a))

		B2 = lambda a : (nc/na)**2*np.sqrt(((na/nc)**2 - np.sin(a)**2)+0j)		

	r2 = lambda a : (np.cos(a)-B2(a))/(np.cos(a)+B2(a))

	phi1 = lambda a : np.angle(r1(a))
	phi2 = lambda a : np.angle(r2(a))

	M = math.floor((4*np.pi*t*nc/lmbda0*np.cos(a0)+phi1(a0) + phi2(a0))/(2*np.pi))

	for m in range(min(kwargs["Modes"],M+1)):
		a = root(lambda a : 4*np.pi*t*nc/lmbda0*np.cos(a)+phi1(a)+phi2(a)-2*(m)*np.pi,1)
		neff.append((np.sin(a.x)*nc)[0])
	return neff		