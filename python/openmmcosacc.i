  
%module openmmcosacc

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_vector.i>

%{
#include "CosAccForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

%feature("shadow") CosAccPlugin::CosAccForce::CosAccForce %{
    def __init__(self, *args):
        this = _openmmcosacc.new_CosAccForce(args[0])
        try:
            self.this.append(this)
        except Exception:
            self.this = this
%}

namespace std {
  %template(IntVector) vector<int>;
}

namespace CosAccPlugin {

class CosAccForce : public OpenMM::Force {
public:
    CosAccForce(double acc);
    void addParticle(double mass);
    void getParticleParameters(int index, double& mass) const;
    void setParticleParameters(int index, double mass);
    double getAcc() const;
    void setAcc(double acc);

    /*
     * Add methods for casting a Force to a DeepMDForce.
    */
    %extend {
        static CosAccPlugin::CosAccForce& cast(OpenMM::Force& force) {
            return dynamic_cast<CosAccPlugin::CosAccForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<CosAccPlugin::CosAccForce*>(&force) != NULL);
        }
    }
};

}