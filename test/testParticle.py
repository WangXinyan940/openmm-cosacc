import simtk.openmm as mm 
import simtk.openmm.app as app
import numpy as np
import simtk.unit as u
import openmmcosacc

neforce = openmmcosacc.CosAccForce(0.25 * u.nanometer / u.picosecond ** 2)