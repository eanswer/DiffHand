#pragma once
#include "Common.h"
#include "Joint/Joint.h"

namespace redmax {

// JointSphericalExp is a spherical joint represented by exponential coordinates
class JointSphericalExp : public Joint {
public:
    JointSphericalExp(Simulation *sim, int id, Joint * parent, Matrix3 R_pj_0, Vector3 p_pj_0,
        Joint::Frame frame = Joint::Frame::LOCAL);

    virtual void update(bool design_gradient = false);

    void inner_update();
    
    bool reparam();

private:
    // auxilary variables
    Matrix3 A[3], B[3], C[3], Adot[3], Bdot[3], Cdot[3];
    RowVector3 dd_dq, dddot_dq;
    JacobianMatrixVector dA_dq[3], dB_dq[3], dC_dq[3], dAdot_dq[3], dBdot_dq[3], dCdot_dq[3];
};

}