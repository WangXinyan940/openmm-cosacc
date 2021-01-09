import simtk.openmm as mm 
import simtk.openmm.app as app
import numpy as np
import simtk.unit as u
import openmmcosacc
import sys

if len(sys.argv) > 1:
    platformName = sys.argv[1]
else:
    platformName = "Reference"

neforce = openmmcosacc.CosAccForce(0.25 * u.nanometer / u.picosecond ** 2)
neforce.setForceGroup(1)

system = mm.System()
for _ in range(8):
    system.addParticle(8.0)

cell = np.array([
    [8.0, 0.0, 0.0],
    [0.0, 8.0, 0.0],
    [0.0, 0.0, 8.0]
]) * u.nanometer

pos = np.zeros((8,3))
for i in range(8):
    pos[i,2] = i

system.setDefaultPeriodicBoxVectors(cell[:,0], cell[:,1], cell[:,2])
system.addForce(neforce)

integ = mm.VerletIntegrator(0.5 * u.femtosecond)
platform = mm.Platform.getPlatformByName(platformName)
ctx = mm.Context(system, integ, platform)
ctx.setPositions(pos)

for i in range(100):
    integ.step(2)
    state = ctx.getState(getPositions=True, getVelocities=True)
    print("Step:", i)
    print("Pos:")
    print(state.getPositions(asNumpy=True))
    print("Vel:")
    print(state.getVelocities(asNumpy=True))