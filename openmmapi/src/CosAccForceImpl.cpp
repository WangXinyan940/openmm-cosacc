#include "internal/CosAccForceImpl.h"
#include "CosAccKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace CosAccPlugin;
using namespace OpenMM;
using namespace std;

CosAccForceImpl::CosAccForceImpl(const CosAccForce& owner) : owner(owner) {
}

CosAccForceImpl::~CosAccForceImpl() {
}

void CosAccForceImpl::initialize(ContextImpl& context) {

    // Create the kernel.
    kernel = context.getPlatform().createKernel(CalcCosAccForceKernel::Name(), context);
    kernel.getAs<CalcCosAccForceKernel>().initialize(context.getSystem(), owner);
}

double CosAccForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcCosAccForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> CosAccForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcCosAccForceKernel::Name());
    return names;
}