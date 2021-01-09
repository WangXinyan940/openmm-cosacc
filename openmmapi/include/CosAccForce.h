#ifndef OPENMM_COSACCFORCE_H_
#define OPENMM_COSACCFORCE_H_

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <string>
#include <vector>

namespace CosAccPlugin {

/**
 * This class implements CosAcc-kit force field. 
 */

class CosAccForce : public OpenMM::Force {
public:
    /**
     * Create a CosAccForce.  The network is defined by a TensorFlow graph saved
     * to a binary protocol buffer file.
     *
     * @param acc        the pre-factor of acceleration
     */
    CosAccForce(double acc);
    /**
     * Get the pre-factor for cos accelerate force.
     */
    double getAcc() const;
    /**
     *  Set the pre-factor for cos accelerate force.
     * 
     * @param acc        the pre-factor
     */
    void setAcc(double acc);
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    double accelerate;
};

} // namespace CosAccPlugin

#endif /*OPENMM_COSACFORCE_H_*/