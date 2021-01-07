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
     * Add atom for CosAcc force field. Mass need to be set to calculate force.
     * 
     * @param mass       the mass of atom
     */
    void addParticle(double mass);
    /**
     * Get the mass for atom i.
     * @param index      the index of atom
     * @param[out] mass  mass of atom
     */
    void getParticleParameters(int index, double& mass) const;
    /**
     * Set mass for atom i.
     * @param index      the index of atom
     * @param mass       the mass of atom
     */
    void setParticleParameters(int index, double mass);
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
    std::vector<double> massvec;
};

} // namespace CosAccPlugin

#endif /*OPENMM_COSACFORCE_H_*/