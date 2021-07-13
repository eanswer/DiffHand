#include "Actuator/ActuatorMotor.h"
#include "Joint/Joint.h"

namespace redmax {

ActuatorMotor::ActuatorMotor(Joint* joint, VectorX ctrl_min, VectorX ctrl_max, std::string name)
    : Actuator(joint->_ndof, ctrl_min, ctrl_max, name) {
    
    _joint = joint;
}

ActuatorMotor::ActuatorMotor(Joint* joint, dtype ctrl_min, dtype ctrl_max, std::string name)
    : Actuator(joint->_ndof, ctrl_min, ctrl_max, name) {
    
    _joint = joint;
}

void ActuatorMotor::computeForce(VectorX& fm, VectorX& fr) {
    VectorX u = _u.cwiseMin(VectorX::Ones(_joint->_ndof)).cwiseMax(VectorX::Ones(_joint->_ndof) * -1.);
    fr.segment(_joint->_index[0], _joint->_ndof) += ((u + VectorX::Ones(_joint->_ndof)) / 2.).cwiseProduct(_ctrl_max - _ctrl_min) + _ctrl_min;
}

void ActuatorMotor::computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr) {
    VectorX u = _u.cwiseMin(VectorX::Ones(_joint->_ndof)).cwiseMax(VectorX::Ones(_joint->_ndof) * -1.);
    fr.segment(_joint->_index[0], _joint->_ndof) += ((u + VectorX::Ones(_joint->_ndof)) / 2.).cwiseProduct(_ctrl_max - _ctrl_min) + _ctrl_min;
}

void ActuatorMotor::compute_dfdu(MatrixX& dfm_du, MatrixX& dfr_du) {
    for (int i = 0;i < _joint->_ndof;i++)
        if (_u[i] < -1. || _u[i] > 1.) {
            dfr_du(_joint->_index[i], _index[i]) += 0.;    
        }
        else {
            dfr_du(_joint->_index[i], _index[i]) += (_ctrl_max[i] - _ctrl_min[i]) / 2.;
        }
}

}