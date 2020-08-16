#%matplotlib notebook
import time, h5py, io, sys
import numpy as np
import matplotlib.pyplot as plt
from lcls_beamline_toolbox.xraybeamline2d import beam1d as beam, optics1d as optics, beamline1d as beamline

# number of sampling points
N = 2048 * 8
# photon energy in eV
E0 = 17795

# parameter dictionary. z_source is in LCLS coordinates (20 meters upstream of undulator exit)
beam_params = {
    'photonEnergy': E0,
    'N': N,
    'sigma_x': 23e-6,
    'sigma_y': 23e-6,
    'rangeFactor': 5,
    'scaleFactor': 20,
    'z_source': 630
}


def define_devices(delta=0, crystal =1, E0=E0):
    # asymmetry angle
    etaA = np.deg2rad(9.4)
    etaB = np.deg2rad(70)
    etaC = np.deg2rad(9.4)
    
    # crystal reflection hkl index
    hklA = [4,0,0]
    hklB = [12,8,4]
    hklC = [4,0,0]
    
    # viewing point upstream of monochromator
    im0 = optics.PPM('im0', z=870, FOV=2e-3, N=256)
    crl0 = optics.CRL('crl0', z=920, E0=E0, f=290., diameter=2e-3)

    # crystal pair A
    crystal1 = optics.Crystal('c1', hkl=hklA, length=10e-2, width=20e-3, z=930., E0=E0,
                              alphaAsym=-etaA, asym_type='emergence', orientation=2, pol='s')

    # second crystal: asymmetric reflection, orientation flipped relative to crystal1
    crystal2 = optics.Crystal('c2', hkl=hklA, length=10e-2, width=20e-3, z=crystal1.z+.2, E0=E0,
                              alphaAsym=-etaA, asym_type='emergence', orientation=0,pol='s')
    
    # viewing point downstream of crystal 2
    im1 = optics.PPM('after_A', z=crystal2.z+.1,N=2048,FOV=16e-3)

    # crystal pair B
    crystal3 = optics.Crystal('c3', hkl=hklB, length=10e-2, width=10e-3, z=crystal2.z+.5, E0=E0,
                              alphaAsym=etaB, asym_type='emergence', orientation=0, pol='s')
    
    crystal4 = optics.Crystal('c4', hkl=hklB, length=10e-2, width=10e-3, z=crystal3.z+.2, E0=E0,
                              alphaAsym=-etaB, asym_type='emergence', orientation=2, pol='s')
    # viewing after crystal 4
    im2 = optics.PPM('after_B', z=crystal4.z+.1,N=2048,FOV=16e-3)

    # crystal pair C
    crystal5 = optics.Crystal('c5', hkl=hklC, length=10e-2, width=10e-3, z=crystal4.z+.5, E0=E0,
                              alphaAsym=etaC, asym_type='emergence', orientation=2, pol='s')
    
    crystal6 = optics.Crystal('c6', hkl=hklC, length=10e-2, width=10e-3, z=crystal5.z+.2, E0=E0,
                              alphaAsym=etaC, asym_type='emergence', orientation=0, pol='s')

    # viewing point just downstream of monochromator
    im3 = optics.PPM('output', z=crystal6.z+.1, FOV=2e-3, N=256)

    # list of devices to propagate through
    devices = [im0, crl0, crystal1, crystal2, im1, crystal3, crystal4, im2, crystal5, crystal6, im3]

    for device in devices:
        if device.name == 'c{}'.format(crystal):
            device.delta = delta
    return devices

# loop (mute print)
trap = io.StringIO()
sys.stdout = trap

tstart = time.time()
# initialize optical elements
devices = define_devices()

# initialize beamline
mono_beamline = beamline.Beamline(devices)
# propagate
beam_params['photonEnergy'] = E0
pulse = beam.Pulse(beam_params=beam_params, tau=400, time_window=20000)
pulse.propagate(beamline=mono_beamline, screen_names=['im0', 'after_A','after_B','output'])
tfin = time.time()

''' energy slices '''
pulse.imshow_energy_slice('im0')
plt.savefig('plots/eV_im0.png')
pulse.imshow_energy_slice('after_A')
plt.savefig('plots/eV_afterA.png')
pulse.imshow_energy_slice('after_B')
plt.savefig('plots/eV_afterB.png')
pulse.imshow_energy_slice('output')
plt.savefig('plots/eV_output.png')

''' spatial projection (x,y)'''
pulse.imshow_projection('im0')
plt.savefig('plots/projection_im0.png')
pulse.imshow_projection('after_A')
plt.savefig('plots/projection_afterA.png')
pulse.imshow_projection('after_B')
plt.savefig('plots/projection_afterB.png')
pulse.imshow_projection('output')
plt.savefig('plots/projection_output.png')

''' time slice '''
print('total {}s, per slice {}ms'.format(round(tfin-tstart,2), round(1000*(tfin-tstart)/pulse.N,2)))

shift = 0
pulse.imshow_time_slice('im0')
plt.savefig('plots/time_im0.png')
pulse.imshow_time_slice('after_A')
plt.savefig('plots/time_afterA.png')
pulse.imshow_time_slice('after_B')
plt.savefig('plots/time_afterB.png')
pulse.imshow_time_slice('output')
plt.savefig('plots/time_output.png')

