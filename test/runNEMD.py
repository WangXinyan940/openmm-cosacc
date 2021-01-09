import simtk.openmm as mm 
import simtk.openmm.app as app
import numpy as np
import simtk.unit as u
import openmmcosacc
import sys
import matplotlib.pyplot as plt

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

neforce = openmmcosacc.CosAccForce(0.025 * u.nanometer / u.picosecond ** 2)
neforce.setForceGroup(0)
system.addForce(neforce)

nbforce = mm.NonbondedForce()
for _ in range(1000):
    nbforce.addParticle(0.0, 0.31507524065751241, 0.635968)
nbforce.setNonbondedMethod(nbforce.CutoffPeriodic)
nbforce.setCutoffDistance(1.0 * u.nanometer)
system.addForce(nbforce)
print("PBC:", system.usesPeriodicBoundaryConditions())

#integ = mm.VerletIntegrator(0.5 * u.femtosecond)
integ = mm.NoseHooverIntegrator(298.15 * u.kelvin, 1.0 * u.picosecond, 0.5 * u.femtosecond)
platform = mm.Platform.getPlatformByName(platformName)
ctx = mm.Context(system, integ, platform)

with open("pos.txt", "r") as f:
    pos = np.array([[float(j) for j in i.strip().split()] for i in f])

with open("vel.txt", "r") as f:
    vel = np.array([[float(j) for j in i.strip().split()] for i in f])

ctx.setPositions(pos)
ctx.setVelocities(vel)

nsample = 100 * 1000
posz = np.zeros((nsample,1000))
velx = np.zeros((nsample,1000))
for step in range(nsample):
    if step % 100 == 0:
        print("Step:", step)
    integ.step(20)
    state = ctx.getState(getPositions=True, getVelocities=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    vel = state.getVelocities(asNumpy=True).value_in_unit(u.nanometer/u.picosecond)
    posz[step,:] = pos[:,2]
    velx[step,:] = vel[:,0]


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


posz = posz.ravel()
velx = velx.ravel()
xbins = np.linspace(0.0, 9.6, 41)
nbins = np.zeros((40,))
totbins = np.zeros((40,))
posz = posz % 9.6
padd = posz // (xbins[1] - xbins[0])

for ii in range(posz.shape[0]):
    if ii%100000 ==0:
        print("%4.2f/100"%(100.0*ii/posz.shape[0]))
    iadd = int(padd[ii])
    nbins[iadd] += 1
    totbins[iadd] += velx[ii]

xaxis = (xbins[:-1] + xbins[1:]) / 2
plt.plot(xaxis, totbins / nbins)
plt.savefig("vx.png")