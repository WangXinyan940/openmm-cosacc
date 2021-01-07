#ifndef OPENMM_CUDA_COSACC_KERNEL_FACTORY_H_
#define OPENMM_CUDA_COSACC_KERNEL_FACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the CUDA implementation of the neural network plugin.
 */

class CudaCosAccKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_CUDA_CosAcc_KERNEL_FACTORY_H_*/
