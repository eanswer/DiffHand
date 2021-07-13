#pragma once
#include "Actuator/Actuator.h"

namespace redmax {

class Joint;

class ActuatorMotor : public Actuator {
public:
    Joint* _joint; // the joint to apply the motor force.

    ActuatorMotor(Joint* joint, VectorX ctrl_min, VectorX ctrl_max, std::string name = "actuator");
    ActuatorMotor(Joint* joint, dtype ctrl_min, dtype ctrl_max, std::string name = "actuator");

    void computeForce(VectorX& fm, VectorX& fr);
    void computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr);

    void compute_dfdu(MatrixX& dfm_du, MatrixX& dfr_du);
};

}