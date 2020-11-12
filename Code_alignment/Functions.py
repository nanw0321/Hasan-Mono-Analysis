import time, h5py, os, sys
import numpy as np
import matplotlib.pyplot as plt
from lcls_beamline_toolbox.xraybeamline2d import beam1d as beam, optics1d as optics, beamline1d as beamline
from lcls_beamline_toolbox.xraybeamline2d.util import Util

''' misc '''
def make_dir(path):
    if not os.path.exists(path):
        print('make path')
        os.mkdir(path)
    else:
        print('path exists')

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

''' define beamline '''
def define_devices(
    f1, f2, slit_width = 1e-3, hkl = [1,1,1], alphaAsym = 0., E0=18e3, f0 = 290., d23=7.):

    # viewing point upstream of monochromator
    im0 = optics.PPM('im0', z=870, FOV=5e-3, N=256)
    crl0 = optics.CRL('crl0', z=920, E0=E0, f=f0, diameter=2e-3)

    # first crystal: symmetric reflection
    crystal1 = optics.Crystal('c1', hkl=hkl, length=10e-2, width=20e-3, z=930, E0=E0,
                              alphaAsym=0, orientation=0, pol='s', delta=0.e-6)

    # second crystal: asymmetric reflection, orientation flipped relative to crystal1
    crystal2 = optics.Crystal('c2', hkl=hkl, length=10e-2, width=20e-3, z=crystal1.z+.2, E0=E0,alphaAsym=alphaAsym, 
                              orientation=2,pol='s', delta=0e-6)
    # printing crystal incidence and reflection angles for confirmation
    print('crystal 2 incidence angle: {:.2f} degrees'.format(crystal2.alpha*180/np.pi))
    print('crystal 2 exit angle: {:.2f} degrees'.format(crystal2.beta0*180/np.pi))

    im_upstream = optics.PPM('im_upstream', z=crystal2.z + f1 - .1, FOV=2e-3, N=256)

    # CRL with ~1 meter focal length (modeled as single element for now)
    crl1 = optics.CRL('crl1', z=crystal2.z+f1, E0=E0, f=f2, diameter=5e-3)

    # viewing point downstream of first crl
    im1 = optics.PPM('im1', z=crl1.z+.1,N=256,FOV=5e-3)

    # slit at focus
    slit = optics.Slit('slit', z=crl1.z+f2, x_width=slit_width, y_width=2e-3)

    # viewing point at focus
    focus = optics.PPM('focus', z=crl1.z+f2 + 1e-3, FOV=5e-3, N=256)

    # second CRL with ~1 meter focal length, for collimation
    crl2 = optics.CRL('crl2', z=crl1.z+2*f2, E0=E0, f=f2, diameter=5e-3)

    # third crystal, symmetric reflection, same orientation as crystal2
    crystal3 = optics.Crystal('c3', hkl=hkl, length=10e-2, width=10e-3, z=crl2.z+d23, E0=E0,alphaAsym=0, orientation=2,
                             asym_type='emergence',pol='s')

    # fourth crystal, asymmetric reflection, same orientation as crystal1
    crystal4 = optics.Crystal('c4', hkl=hkl, length=10e-2, width=10e-3, z=crl2.z+d23 + (f1-d23)*np.cos(crystal1.beta0*2), E0=E0,alphaAsym=-alphaAsym, 
                              asym_type='emergence', orientation=0,pol='s')

    # viewing point just downstream of monochromator
    im2 = optics.PPM('im2', z=crystal4.z+.1, FOV=5e-3, N=256)

    # list of devices to propagate through
    devices = [crl0,im0,crystal1,crystal2,im_upstream,im1,crl1,slit,focus,crl2,crystal3,crystal4,im2]

    return devices

def change_delta(devices, delta, crystal):
    for device in devices:
        if device.name == 'c{}'.format(crystal):
            device.delta = delta

def shift_z(mono_beamline, shift, oe):
    for device in mono_beamline.device_list:
        if device.name == oe:
            device.z += shift

def change_miscut(devices, eta_err, crystal):
    for i, device in enumerate(devices):
        if device.name == 'c{}'.format(crystal):
            devices[i] = optics.Crystal(device.name, hkl=device.hkl, length=device.length, width=device.width,
                                        z=device.z, E0=device.E0, alphaAsym=device.alphaAsym+eta_err,
                                        orientation=device.orientation, pol=device.pol, delta=device.delta)

def add_shapeError(devices, shapeError, crystal):
    for i, device in enumerate(devices):
        if device.name == 'c{}'.format(crystal):
            devices[i] = optics.Crystal(device.name, hkl=device.hkl, length=device.length, width=device.width,
                                        z=device.z, E0=device.E0, alphaAsym=device.alphaAsym,
                                        orientation=device.orientation, pol=device.pol, delta=device.delta,
                                        shapeError = shapeError)

def lens_energyError(devices, E):
	for i, device in enumerate(devices):
		if device.name[:3] == 'crl':
			devices[i] = optics.CRL('crl1', z=device.z, E0=E, f=device.f, diameter=device.diameter)

''' get info '''
def print_oe_pos(oe):
    print('{}, x:{}, y:{}, z:{}'.format(oe.name, oe.global_x, oe.global_y, oe.z))
    return oe.global_x, oe.global_y, oe.z

def get_pulse(pulse, image_name, x_pos=0, y_pos=0, shift=None):
    minx = np.round(np.min(pulse.x[image_name]) * 1e6)
    maxx = np.round(np.max(pulse.x[image_name]) * 1e6)
    miny = np.round(np.min(pulse.y[image_name]) * 1e6)
    maxy = np.round(np.max(pulse.y[image_name]) * 1e6)

    # get number of pixels
    M = pulse.x[image_name].size
    N = pulse.y[image_name].size

    # calculate pixel sizes (microns)
    dx = (maxx - minx) / M
    dy = (maxy - miny) / N

    # calculate indices for the desired location
    x_index = int((x_pos - minx) / dx)
    y_index = int((y_pos - miny) / dy)

    # calculate temporal intensity
    y_data = np.abs(pulse.time_stacks[image_name][y_index, x_index, :]) ** 2

    shift = -pulse.t_axis[np.argmax(y_data)]

    # coarse shift for fitting
    if shift is not None:
        y_data = np.roll(y_data, int(shift/pulse.deltaT))

    # get gaussian stats
    centroid, sx = Util.gaussian_stats(pulse.t_axis, y_data)
    fwhm = int(sx * 2.355)

    # gaussian fit
    gauss_plot = Util.fit_gaussian(pulse.t_axis, centroid, sx)

    # shift again using fit result
    shift = -centroid
    if shift is not None:
        y_data = np.roll(y_data, int(shift/pulse.deltaT))
        gauss_plot = np.roll(gauss_plot, int(shift/pulse.deltaT))
        
    # [fs], normalized intensity [simulated], [Gaussian Fit]
    return pulse.t_axis, y_data/np.max(y_data), gauss_plot

def get_spectrum(pulse, image_name, x_pos=0, y_pos=0, integrated=False):
    minx = np.round(np.min(pulse.x[image_name]) * 1e6)
    maxx = np.round(np.max(pulse.x[image_name]) * 1e6)
    miny = np.round(np.min(pulse.y[image_name]) * 1e6)
    maxy = np.round(np.max(pulse.y[image_name]) * 1e6)

    # get number of pixels
    M = pulse.x[image_name].size
    N = pulse.y[image_name].size

    # calculate pixel sizes (microns)
    dx = (maxx - minx) / M
    dy = (maxy - miny) / N

    # calculate indices for the desired location
    x_index = int((x_pos - minx) / dx)
    y_index = int((y_pos - miny) / dy)

    # calculate spectral intensity
    if integrated:
        y_data = np.sum(np.abs(pulse.energy_stacks[image_name])**2, axis=(0,1))
    else:
        y_data = np.abs(pulse.energy_stacks[image_name][y_index,x_index,:])**2

    # get gaussian stats
    centroid, sx = Util.gaussian_stats(pulse.energy, y_data)
    fwhm = sx * 2.355

    # gaussian fit to plot
    gauss_plot = Util.fit_gaussian(pulse.energy, centroid, sx)

    # change label depending on bandwidth
    if fwhm >= 1:
        width_label = '%.1f eV FWHM' % fwhm
    elif fwhm > 1e-3:
        width_label = '%.1f meV FHWM' % (fwhm * 1e3)
    else:
        width_label = u'%.1f \u03BCeV FWHM' % (fwhm * 1e6)
    
    # [eV], normalized intensity [simulated], [Gaussian Fit]
    return pulse.energy - pulse.E0, y_data/np.max(y_data), gauss_plot