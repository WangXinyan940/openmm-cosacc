  
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

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
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
        if unit.is_quantity(args[0]):
            acc = args[0].value_in_unit(unit.nanometer / unit.picosecond ** 2)
        else:
            acc = args[0]
        this = _openmmcosacc.new_CosAccForce(acc)
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