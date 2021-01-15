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
        double masstmp = system.getParticleMass(i);
        massvec[i] = masstmp > force.getLimit() ? masstmp: 0.0;
    }
    accelerate = force.getAcc();
}

double ReferenceCalcCosAccForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& force = extractForces(context);
    Vec3* box = extractBoxVectors(context);
    double oneLz = 1.0/box[2][2];
    int numParticles = pos.size();
    double energy = 0.0;
    if (includeEnergy){
        energy += 1.0;
    }
    if (includeForces){
        for(int i=0; i<numParticles; i++){
            double addfrc = accelerate * cos(6.283185307179586*pos[i][2]*oneLz) * massvec[i];
            force[i][0] += addfrc;
        }
    }
    return energy;
}