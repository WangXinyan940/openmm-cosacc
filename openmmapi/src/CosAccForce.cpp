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

double CosAccForce::getAcc() const{
    return accelerate;
}

void CosAccForce::setAcc(double acc){
    accelerate = acc;
}

ForceImpl* CosAccForce::createImpl() const {
    return new CosAccForceImpl(*this);
}

