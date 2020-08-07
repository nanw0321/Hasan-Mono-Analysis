import numpy as np
import os, copy, h5py

# beam path calculation
def calc_Bragg(material='Si', h=1, k=1, l=1, w0=1e-10):
	if material == 'Si':
		a = 5.431020511e-10,0,0
	if material == 'Ge':
		a = 5.68e-10,0,0
	d_hkl = a/np.linalg.norm(h**2+k**2+l**2)	# interplanary distance
	theta_B = np.arcsin(w0/2/d_hkl)				# Bragg angle in radian
	return theta_B

def calc_wavelength(material='Si', h=1, k=1, l=1, theta_B):
	if material == 'Si':
		a = 5.431020511e-10,0,0
	if material == 'Ge':
		a = 5.68e-10,0,0
	d_hkl = a/np.linalg.norm(h**2+k**2+l**2)	# interplanary distance
	w0 = np.sin(theta_B)*2*d_hkl
	return w0