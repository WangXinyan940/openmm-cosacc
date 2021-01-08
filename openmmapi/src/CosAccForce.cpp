#include "CosAccForce.h"
#include "internal/CosAccForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>

using namespace CosAccPlugin;
using namespace OpenMM;
using namespace std;

CosAccForce::CosAccForce(double acc) : accelerate(acc) {
}

void CosAccForce::addParticle(double mass){
    massvec.push_back(mass);
}

void CosAccForce::getParticleParameters(int index, double& mass) const {
    mass = massvec[index];
}

void CosAccForce::setParticleParameters(int index, double mass){
    massvec[index] = mass;
}

double CosAccForce::getAcc() const{
    return accelerate;
}

void CosAccForce::setAcc(double acc){
    accelerate = acc;
}

ForceImpl* CosAccForce::createImpl() const {
    return new CosAccForceImpl(*this);
}

