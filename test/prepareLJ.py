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

system = mm.System()
for _ in range(1000):
    system.addParticle(16.0)

cell = np.array([
    [2.4, 0.0, 0.0],
    [0.0, 2.4, 0.0],
    [0.0, 0.0, 9.6]
]) * u.nanometer
system.setDefaultPeriodicBoxVectors(cell[:,0], cell[:,1], cell[:,2])

nbforce = mm.NonbondedForce()
for _ in range(1000):
    nbforce.addParticle(0.0, 0.31507524065751241, 0.635968)
nbforce.setNonbondedMethod(nbforce.CutoffPeriodic)
nbforce.setCutoffDistance(1.0 * u.nanometer)
system.addForce(nbforce)

integ = mm.LangevinIntegrator(298.15 * u.kelvin, 1.0 / u.picosecond, 0.5 * u.femtosecond)
platform = mm.Platform.getPlatformByName(platformName)
ctx = mm.Context(system, integ, platform)
pos = np.random.random((1000,3)) * np.array([[2.4, 2.4, 9.6]])
ctx.setPositions(pos)
mm.LocalEnergyMinimizer.minimize(ctx)
for loop in range(1000):
    print(loop)
    integ.step(100)
state = ctx.getState(getPositions=True, getVelocities=True)
pos = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
vel = state.getVelocities(asNumpy=True).value_in_unit(u.nanometer/u.picosecond)
with open("pos.txt" , "w") as f:
    for x, y, z in pos:
        f.write("%16.8f%16.8f%16.8f\n"%(x, y, z))
with open("vel.txt" , "w") as f:
    for x, y, z in vel:
        f.write("%16.8f%16.8f%16.8f\n"%(x, y, z))