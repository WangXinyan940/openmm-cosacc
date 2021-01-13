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
    int elementSize = cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float);

    // create input tensor
    massvec_cu.initialize(cu, numParticles, elementSize, "massvec_cu");
    if (cu.getUseDoublePrecision()){
        vector<double> massvec;
        massvec.resize(numParticles);
        for(int i=0;i<numParticles;i++){
            massvec[i] = system.getParticleMass(i);
        }
        massvec_cu.upload(massvec);
    } else {
        vector<float> massvec;
        massvec.resize(numParticles);
        for(int i=0;i<numParticles;i++){
            massvec[i] = system.getParticleMass(i);
        }
        massvec_cu.upload(massvec);
    }

    accelerate = force.getAcc();

    // Inititalize CUDA objects.
    map<string, string> defines;

    Vec3 boxVectors[3];
    system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    defines["TWOPIOVERLZ"] = cu.doubleToString(6.283185307179586/boxVectors[2][2]);
    cout << "TWOPIOVERLZ: " << defines["TWOPIOVERLZ"] << endl;
    CUmodule module = cu.createModule(CudaCosAccKernelSources::cosAccForce, defines);
    addForcesKernel = cu.getKernel(module, "addForces");
    hasInitializedKernel = true;
}

double CudaCalcCosAccForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    
    int numParticles = cu.getNumAtoms();

    double energy = 0.0;
    if (includeEnergy) {
        energy += 1.0;
    }
    if (includeForces) {
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&massvec_cu.getDevicePointer(), &cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &accelerate, &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }
    return energy;
}