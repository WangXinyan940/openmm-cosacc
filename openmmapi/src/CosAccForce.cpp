#include "CosAccForce.h"
#include "internal/CosAccForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>

using namespace CosAccPlugin;
using namespace OpenMM;
using namespace std;

CosAccForce::CosAccForce(double acc, double dt) : acceleration(acc), timestep(dt) {
}

double CosAccForce::getAcc() const{
    return acceleration;
}

void CosAccForce::setAcc(double acc){
    acceleration = acc;
}

double CosAccForce::getTimestep() const{
    return timestep;
}

void CosAccForce::setTimestep(double dt){
    timestep = dt;
}

ForceImpl* CosAccForce::createImpl() const {
    return new CosAccForceImpl(*this);
}

