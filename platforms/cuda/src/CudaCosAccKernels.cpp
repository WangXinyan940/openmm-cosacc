#include "CudaCosAccKernels.h"
#include "CudaCosAccKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <iostream>

using namespace CosAccPlugin;
using namespace OpenMM;
using namespace std;

CudaCalcCosAccForceKernel::~CudaCalcCosAccForceKernel() {
}

void CudaCalcCosAccForceKernel::initialize(const System& system, const CosAccForce& force) {
    cu.setAsCurrent();

    int numParticles = system.getNumParticles();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));

    // create input tensor
    for(int i=0;i<numParticles;i++){
        double masstmp;
        force.getParticleParameters(i, &masstmp);
        massvec.push_back(masstmp);
    }

    massvec_cu.initialize(cu, numParticles, elementSize, "massvec_cu");
    massvec_cu.upload(massvec);

    accelerate = force.getAcc();

    // Inititalize CUDA objects.
    map<string, string> defines;
    if (cu.getUseDoublePrecision()){
        defines["FORCES_TYPE"] = "double";
    } else {
        defines["FORCES_TYPE"] = "float";
    }

    Vec3 boxVectors[3];
    system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    defines["2PIONELZ"] = cu.doubleToString(2.0*3.141592654/boxVectors[2][2]);
    
    CUmodule module = cu.createModule(CudaCosAccKernelSources::cosAccForce, defines);
    addForcesKernel = cu.getKernel(module, "addForces");
}

double CudaCalcCosAccForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    
    int numParticles = cu.getNumAtoms();

    double energy = 0.0;
    if (includeEnergy) {
        energy += 1.0;
    }
    if (includeForces) {
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&massvec_cu.getDevicePointer(), &cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &accelerate, &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }
    return energy;
}