import numpy as np 
import matplotlib.pyplot as plt

with open("posz.txt", "r") as f:
    posz = np.array([[float(j) for j in i.strip().split()] for i in f]).ravel()

with open("velx.txt", "r") as f:
    velx = np.array([[float(j) for j in i.strip().split()] for i in f]).ravel()

xbins = np.linspace(0.0, 2.4, 41)
nbins = np.zeros((40,))
totbins = np.zeros((40,))

for ii in range(posz.shape[0]):
    if ii%10000 ==0:
        print("%4.2f/100"%(100.0*ii/posz.shape[0]))
    tempz = posz[ii] % 2.4
    for n in range(40):
        iadd = tempz % (xbins[1] - xbins[0])
        if iadd < 40:
            nbins[iadd] += 1
            totbins[iadd] += velx[ii]

xaxis = (xbins[:-1] + xbins[1:]) / 2
plt.plot(xaxis, totbins / nbins)
plt.savefig("vx.png")