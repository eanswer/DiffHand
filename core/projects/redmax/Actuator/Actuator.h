#pragma once
#include "Common.h"
#include "Utils.h"

namespace redmax {

class Actuator {
public:
    std::string _name;
    int _ndof;                      // number of action dofs
    std::vector<int> _index;        // the indices in the control vector u
    VectorX _u;                     // control signals [-1, 1]
    VectorX _ctrl_min, _ctrl_max;   // control range min / max

    Actuator(int ndof, VectorX ctrl_min, VectorX ctrl_max, std::string name = "actuator");
    Actuator(int ndof, dtype ctrl_min, dtype ctrl_max, std::string name = "actuator");

    void get_ctrl_range(VectorX& ctrl_min, VectorX& ctrl_max);

    void set_u(const VectorX& u);

    virtual void computeForce(VectorX& fm, VectorX& fr) = 0;
    virtual void computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr) = 0;

    virtual void compute_dfdu(MatrixX& dfm_du, MatrixX& dfr_du) = 0;
};

}
