#ifndef REFERENCE_COSACC_KERNELS_H_
#define REFERENCE_COSACC_KERNELS_H_

#include "CosAccKernels.h"
#include "openmm/Platform.h"
#include <vector>
#include <set>
#include <iostream>

namespace CosAccPlugin {

/**
 * This kernel is invoked by CosAccForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcCosAccForceKernel : public CalcCosAccForceKernel {
public:
    ReferenceCalcCosAccForceKernel(std::string name, const OpenMM::Platform& platform) : CalcCosAccForceKernel(name, platform) {
    }
    ~ReferenceCalcCosAccForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system         the System this kernel will be applied to
     * @param force          the CosAccForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const CosAccForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    double accelerate;
    vector<double> massvec;
};

} // namespace CosAccPlugin

#endif /*REFERENCE_COSACC_KERNELS_H_*/