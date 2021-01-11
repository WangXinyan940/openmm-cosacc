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
    //massvec_cu.initialize(cu, numParticles, elementSize, "massvec_cu");
    //if (cu.getUseDoublePrecision()){
    //    vector<double> massvec;
    //    massvec.resize(numParticles);
    //    for(int i=0;i<numParticles;i++){
    //        massvec[i] = system.getParticleMass(i);
    //    }
    //    massvec_cu.upload(massvec);
    //} else {
    //    vector<float> massvec;
    //    massvec.resize(numParticles);
    //    for(int i=0;i<numParticles;i++){
    //        massvec[i] = system.getParticleMass(i);
    //    }
    //    massvec_cu.upload(massvec);
    //}

    acceleration = force.getAcc() * force.getTimestep();

    // Inititalize CUDA objects.
    map<string, string> defines;
    //if (cu.getUseDoublePrecision()){
    //    defines["FORCES_TYPE"] = "double";
    //} else {
    //    defines["FORCES_TYPE"] = "float";
    //}

    Vec3 boxVectors[3];
    system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    defines["TWOPIOVERLZ"] = cu.doubleToString(6.283185307179586/boxVectors[2][2]);
    
    CUmodule module = cu.createModule(CudaCosAccKernelSources::cosAccForce, defines);
    addVelsKernel = cu.getKernel(module, "addVels");
    hasInitializedKernel = true;
}

void CudaCalcCosAccForceKernel::execute(ContextImpl& context) {
    int numParticles = cu.getNumAtoms();
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    void* args[] = {&cu.getPosq().getDevicePointer(), &cu.getVelm().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &acceleration, &numParticles, &paddedNumAtoms};
    cu.executeKernel(addVelsKernel, args, numParticles);
}