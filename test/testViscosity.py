import simtk.openmm as mm 
import simtk.openmm.app as app
import numpy as np
import simtk.unit as u
import openmmcosacc

if len(sys.argv) > 1:
    platformName = sys.argv[1]
else:
    platformName = "Reference"

system = mm.System()
for _ in range(250):
    system.addParticle(16.0)

cell = np.array([
    [2.4, 0.0, 0.0],
    [0.0, 2.4, 0.0],
    [0.0, 0.0, 2.4]
]) * u.nanometer
system.setDefaultPeriodicBoxVectors(cell[:,0], cell[:,1], cell[:,2])

neforce = openmmcosacc.CosAccForce(0.25 * u.nanometer / u.picosecond ** 2)
neforce.setForceGroup(1)
system.addForce(neforce)

nbforce = mm.NonbondedForce()
for _ in range(250):
    nbforce.addParticle(0.0, 0.31507524065751241, 0.635968)
nbforce.setNonbondedMethod(nbforce.CutoffPeriodic)
nbforce.setCutoffDistance(1.0 * u.nanometer)
system.addForce(nbforce)

