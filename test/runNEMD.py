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

integ = mm.NoseHooverIntegrator(298.15 * u.kelvin, 1.0 * u.picosecond, 0.5 * u.femtosecond)
platform = mm.Platform.getPlatformByName(platformName)
ctx = mm.Context(system, integ, platform)

with open("pos.txt", "r") as f:
    pos = np.array([[float(j) for j in i.strip().split()] for i in f])

with open("vel.txt", "r") as f:
    vel = np.array([[float(j) for j in i.strip().split()] for i in f])

ctx.setPositions(pos)
ctx.setVelocities(vel)
posz = []
velx = []
for step in range(100 * 1000):
    if step % 100 == 0:
        print("Step:", step)
    integ.step(2)
    state = ctx.getState(getPositions=True, getVelocities=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    vel = state.getVelocities(asNumpy=True).value_in_unit(u.nanometer/u.picosecond)
    posz.append(pos[:,2])
    velx.append(vel[:,0])

with open("posz.txt", "w") as f:
    for i in posz:
        for j in i:
            f.write("%16.8f"%j)
        f.write("\n")

with open("velx.txt", "w") as f:
    for i in velx:
        for j in i:
            f.write("%16.8f"%j)
        f.write("\n")