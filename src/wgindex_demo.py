import numpy as np


def wgindex(lmbda0,w,h,t,na,nc,ns,**kwargs):
	"""Effective index method for guided modes in arbitrary waveguide
	
	DESCRIPTION:
	  solves for the TE (or TM) effective index of an etched waveguide
	  structure using the effectice index method.
	
	USAGE:
	  - get effective index for supported TE-like modes:
	  neff = eim_index(1.55, 0.5, 0.22, 0.09, 1, 3.47, 1.44)
	
	             |<   w   >|
	              _________           _____
	             |         |            ^
	 ___    _____|         |_____ 
	  ^                                 h
	  t                                  
	 _v_    _____________________     __v__
	
	         II  |    I    |  II
	
	INPUT:
	lambda0   - free-space wavelength
	w         - core width
	h         - slab thickness
	t         - slab thickness
	              t < h  : rib waveguide
	              t == 0 : rectangular waveguide w x h
	              t == h : uniform slab of thickness t
	na        - (top) oxide cladding layer material index
	nc        - (middle) core layer material index
	ns        - (bottom) substrate layer material index
	
	OPTIONS:
	Modes - number of modes to solve
	Polarisation - one of 'TE' or 'TM'
	
	OUTPUT:
	neff - TE (or TM) mode index (array of index if multimode)
	
	NOTE: it is possible to provide a function of the form n = lambda lambda0: func(lambda0) for 
	the refractive index which will be called using lambda0."""


	_in = kwargs
	if "Modes" not in _in.keys():
		_in["Modes"] = np.inf

	if "Polarisation" not in _in.keys():
		_in["Polarisation"] = "TE"

	if (type(na) == types.FunctionType) or (str(type(na)) == "<class 'material.Material.Material'>"):
		na = na(lmbda0)
	if (type(nc) == types.FunctionType) or (str(type(nc)) == "<class 'material.Material.Material'>"):
		nc = nc(lmbda0)
	if (type(ns) == types.FunctionType) or (str(type(ns)) == "<class 'material.Material.Material'>"):
		ns = ns(lmbda0)

	t = clamp(t,0,h)

	neff_I = slabindex(lmbda0,h,na,nc,ns,Modes = _in["Modes"], Polarisation = _in["Polarisation"])

	if t == h:
		return neff_I
	if t > 0:
		neff_II = slabindex(lmbda0,t,na,nc,ns,Modes = _in["Modes"],Polarisation = _in["Polarisation"])
	else:
		neff_II = na

	neff = []

	if _in["Polarisation"].upper() in "TE":

		for m in range(min(len(neff_I),len(neff_II))):
			n = slabindex(lmbda0,w,neff_II[m],neff_I[m],neff_II[m],Modes = _in["Modes"],Polarisation = "TM")
			neff.extend(i for i in n if i > max(ns,na))
	else:
		for m in range(min(len(neff_I),len(neff_II))):
			n = slabindex(lmbda0,w,neff_II[m],neff_I[m],neff_II[m],Modes = _in["Modes"],Polarisation = "TE")
			neff.extend(i for i in n if i > max(ns,na))
	return neff