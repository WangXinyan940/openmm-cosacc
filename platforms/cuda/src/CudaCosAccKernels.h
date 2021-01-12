#ifndef CUDA_COSACC_KERNELS_H_
#define CUDA_COSACC_KERNELS_H_

#include "CosAccKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>
#include <string>

namespace CosAccPlugin {

/**
 * This kernel is invoked by CosAccForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcCosAccForceKernel : public CalcCosAccForceKernel {
public:
    CudaCalcCosAccForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu) :
            CalcCosAccForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    }
    ~CudaCalcCosAccForceKernel();
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
    bool hasInitializedKernel;
    double accelerate;
    OpenMM::CudaArray massvec_cu;
    OpenMM::CudaContext& cu;
    CUfunction addForcesKernel;
};

} // namespace CosAccPlugin

#endif /*CUDA_COSACC_KERNELS_H_*/