#include "ReferenceCosAccKernels.h"
#include "CosAccForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include <math.h>

using namespace OpenMM;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->positions);
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
    for(int i=0;i<numParticles;i++){
        massvec.push_back(force.massvec[i]);
    }
    accelerate = force.accelerate;
}

double ReferenceCalcCosAccForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& force = extractForces(context);
    Vec3* box = extractBoxVectors(context);
    double oneLz = box[2][2];
    int numParticles = pos.size();
    for(int i=0; i<numParticles; i++){
        double addfrc = accelerate * cos(6.283185307179586*pos[i][2]*oneLz);
        force[i][0] += addfrc;
    }
    return 1.0;
}