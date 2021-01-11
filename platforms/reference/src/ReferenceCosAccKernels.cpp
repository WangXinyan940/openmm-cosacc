#include "ReferenceCosAccKernels.h"
#include "CosAccForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include <math.h>

using namespace OpenMM;
using namespace std;
using namespace CosAccPlugin;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->positions);
}

static vector<Vec3>& extractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->velocities;
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->forces);
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

ReferenceCalcCosAccForceKernel::~ReferenceCalcCosAccForceKernel() {
}

void ReferenceCalcCosAccForceKernel::initialize(const System& system, const CosAccForce& force) {
    int numParticles = system.getNumParticles();
    massvec.resize(numParticles);
    for(int i=0;i<numParticles;i++){
        massvec[i] = system.getParticleMass(i);
    }
    acceleration = force.getAcc() * force.getTimestep();
}

void ReferenceCalcCosAccForceKernel::execute(ContextImpl& context) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& vels = extractVelocities(context);
    Vec3* box = extractBoxVectors(context);
    double oneLz = 1.0/box[2][2];
    int numParticles = pos.size();
    for(int i=0; i<numParticles; i++){
        if (massvec[i] > 1e-4) {
            double addvel = acceleration * cos(6.283185307179586*pos[i][2]*oneLz);
            vels[i][0] += addvel;
        }
    }
}