#include "Simulation.h"
#include "Robot.h"
#include "Body/Body.h"
#include "Joint/Joint.h"
#include "SimViewer.h"
#include "Force/Force.h"
#include "Force/ForceGeneralPrimitiveContact.h"
#include "VirtualObject/VirtualObject.h"
#include "Sensor/TactileSensor.h"
#include "EndEffector/EndEffector.h"

namespace redmax {

Simulation::~Simulation() {
}

/******************************************* Forward Dynamics ************************************************/

// init simulation
void Simulation::init(bool verbose) {
    _robot->init(verbose);

    _ndof_r = _robot->_ndof_r;
    _ndof_m = _robot->_ndof_m;
    _ndof_u = _robot->_ndof_u;
    _ndof_var = _robot->_ndof_var;
    _ndof_tactile = _robot->_ndof_tactile;
    _ndof_p = _robot->_ndof_p;
    _ndof_p1 = _robot->_ndof_p1;
    _ndof_p2 = _robot->_ndof_p2;
    _ndof_p3 = _robot->_ndof_p3;
    _ndof_p4 = _robot->_ndof_p4;
    _ndof_p5 = _robot->_ndof_p5;
    _ndof_p6 = _robot->_ndof_p6;

    // reserve memory for matrices
    _J = MatrixX::Zero(_ndof_m, _ndof_r);
    _Jdot = MatrixX::Zero(_ndof_m, _ndof_r);
    _dJ_dq = JacobianMatrixVector(_ndof_m, _ndof_r, _ndof_r);
    _dJdot_dq = JacobianMatrixVector(_ndof_m, _ndof_r, _ndof_r);
    _Mm = VectorX::Zero(_ndof_m);
    _fm = VectorX::Zero(_ndof_m);
    _fr = VectorX::Zero(_ndof_r);
    _Km = MatrixX::Zero(_ndof_m, _ndof_m);
    _Dm = MatrixX::Zero(_ndof_m, _ndof_m);
    _Kr = MatrixX::Zero(_ndof_r, _ndof_r);
    _Dr = MatrixX::Zero(_ndof_r, _ndof_r);
    _dfm_du = MatrixX::Zero(_ndof_m, _ndof_u);
    _M = MatrixX::Zero(_ndof_r, _ndof_r);
    _K = MatrixX::Zero(_ndof_r, _ndof_r);
    _D = MatrixX::Zero(_ndof_r, _ndof_r);
    _dfr_du = MatrixX::Zero(_ndof_r, _ndof_u);
    _dphi_dq = MatrixX::Zero(_ndof_m, _ndof_r);
    _dM_dq = JacobianMatrixVector(_ndof_r, _ndof_r, _ndof_r);
    _dM_dp = JacobianMatrixVector(_ndof_r, _ndof_r, _ndof_p);
    _dfr_dp = MatrixX::Zero(_ndof_r, _ndof_p);
    _dfr_dqprev = MatrixX::Zero(_ndof_r, _ndof_r);
    _dfr_dqdotprev = MatrixX::Zero(_ndof_r, _ndof_r);

    // q_init and qdot_init
    _q_init = get_q();
    _qdot_init = get_qdot();
}

// init states
void Simulation::set_state_init(const VectorX q_init, const VectorX qdot_init) {
    set_q_init(q_init);
    set_qdot_init(qdot_init);
}

void Simulation::set_q_init(const VectorX q_init) {
    if (q_init.size() != _ndof_r) {
        std::cerr << "[Error] set_q_init: q_init.size() != _ndof_r." << std::endl;
        throw "error";
    }
    _q_init = q_init;
}

void Simulation::set_qdot_init(const VectorX qdot_init) {
    if (qdot_init.size() != _ndof_r) {
        std::cerr << "[Error] set_qdot_init: qdot_init.size() != _ndof_r." << std::endl;
        throw "error";
    }
    _qdot_init = qdot_init;
}

const VectorX Simulation::get_q_init() {
    return _q_init;
}

const VectorX Simulation::get_qdot_init() {
    return _qdot_init;
}

// states
void Simulation::set_state(const VectorX q, const VectorX qdot) {
    set_q(q);
    set_qdot(qdot);
}

void Simulation::set_q(const VectorX q) {
    _robot->set_q(q);
}

void Simulation::set_qdot(const VectorX qdot) {
    _robot->set_qdot(qdot);
}

const VectorX Simulation::get_q() {
    return _robot->get_q();
}

const VectorX Simulation::get_qdot() {
    return _robot->get_qdot();
}

void Simulation::reparam() {
    _robot->reparam();
}

const VectorX Simulation::get_variables() {
    return _robot->get_variables();
}

// control variables
void Simulation::set_u(const VectorX& u) {
    if (u.size() != _ndof_u) {
        std::cerr << "[Error] set_u: u.size() != _ndof_u." << std::endl;
        throw "error";
    }
    _robot->set_u(u);
}

void Simulation::get_ctrl_range(VectorX& ctrl_min, VectorX& ctrl_max) {
    _robot->get_ctrl_range(ctrl_min, ctrl_max);
}

VectorX Simulation::get_ctrl_force() {
    return _robot->get_ctrl_force();
}

void Simulation::print_ctrl_info() {
    _robot->print_ctrl_info();
}

/** tactile related functions **/

// get parameters
std::vector<Vector3> Simulation::get_tactile_sensor_pos(std::string name) {
    return _robot->get_tactile_sensor_pos(name);
}

std::vector<Vector2i> Simulation::get_tactile_image_pos(std::string name) {
    return _robot->get_tactile_image_pos(name);
}

std::vector<dtype> Simulation::get_tactile_depth(std::string name) {
    return _robot->get_tactile_depth(name);
}

std::vector<dtype> Simulation::get_tactile_normal_force(std::string name) {
    return _robot->get_tactile_normal_force(name);
}

std::vector<Vector2> Simulation::get_tactile_shear_force(std::string name) {
    return _robot->get_tactile_shear_force(name);
}

std::vector<Vector3> Simulation::get_tactile_force(const string name) {
    return _robot->get_tactile_force(name);
}

VectorX Simulation::get_tactile_force_vector() {
    return _robot->get_tactile_force_vector();
}

std::vector<std::vector<std::vector<Vector3>>> Simulation::get_tactile_flow_images() {
    return _robot->get_tactile_flow_images();
}

// design parameters
void Simulation::set_design_params(const VectorX &design_params) {
    _robot->set_design_params(design_params);
}  

VectorX Simulation::get_design_params() {
    return _robot->get_design_params();
}

void Simulation::print_design_params_info() {
    if (_ndof_p > 0)
        _robot->print_design_params_info();
}

// set contact coefficient scale
void Simulation::set_contact_scale(dtype scale) {
    _robot->set_contact_scale(scale);
}

void Simulation::set_rendering_mesh_vertices(const std::vector<Matrix3X> &Vs) {
    _robot->set_rendering_mesh_vertices(Vs);
}

void Simulation::set_rendering_mesh(const std::vector<Matrix3X> &Vs, const std::vector<Matrix3Xi> &Fs) {
    _robot->set_rendering_mesh(Vs, Fs);
}

// virtual objects
void Simulation::update_virtual_object(std::string name, VectorX data) {
    _robot->update_virtual_object(name, data);
}

// update robot
void Simulation::update_robot(bool design_gradient) {
    _robot->update(design_gradient);
}

// functions to update simulation parameters
void Simulation::update_contact_parameters(std::string body1, std::string body2, dtype kn, dtype kt, dtype mu, dtype damping) {
    _robot->update_contact_parameters(body1, body2, kn, kt, mu, damping);
}

void Simulation::update_tactile_parameters(std::string name, dtype kn, dtype kt, dtype mu, dtype damping) {
    _robot->update_tactile_parameters(name, kn, kt, mu, damping);
}

void Simulation::update_body_density(std::string body_name, dtype density) {
    _robot->update_body_density(body_name, density);
}

void Simulation::update_body_color(std::string body_name, Vector3 color) {
    _robot->update_body_color(body_name, color);
}

void Simulation::update_body_size(std::string body_name, VectorX body_size) {
    _robot->update_body_size(body_name, body_size);
}

void Simulation::update_joint_damping(std::string joint_name, dtype damping) {
    _robot->update_joint_damping(joint_name, damping);
}

void Simulation::update_tactile_sensor_pos(std::string name, std::vector<Vector3> &new_pos) {
    _robot->update_tactile_sensor_pos(name, new_pos);
}

void Simulation::update_joint_location(std::string joint_name, Vector3 joint_location) {
    _robot->update_joint_location(joint_name, joint_location);
}

void Simulation::update_endeffector_position(std::string endeffector_name, Vector3 position) {
    _robot->update_endeffector_position(endeffector_name, position);
}

// compute the matrices M, f, so that M*qddot = f
void Simulation::computeMatrices(MatrixX& M, VectorX& fr) {
    auto t_compute_matrices_start = clock();

    _robot->update();

    VectorX qdot = get_qdot();

    _robot->computeJointJacobian(_J, _Jdot);

    _robot->computeMaximalMassMatrix(_Mm);

    auto t_compute_force_start = clock();
 
    _robot->computeForce(_fm, _fr);

    _time_report._time_compute_force += clock() - t_compute_force_start;

    _time_report._time_compute_matrices += clock() - t_compute_matrices_start;

    auto t_compose_matrices_start = clock();

    MatrixX JT = _J.transpose();
    
    MatrixX MmJ(_ndof_m, _ndof_r);
    for (int i = 0;i < MmJ.rows();i++) 
        MmJ.row(i).noalias() = _J.row(i) * _Mm(i);

    MatrixX MmJdot(_ndof_m, _ndof_r);
    for (int i = 0;i < MmJdot.rows();i++)
        MmJdot.row(i).noalias() = _Jdot.row(i) * _Mm(i);

    M = JT * MmJ;
    fr = JT * (_fm - MmJdot * qdot) + _fr;

    _time_report._time_compose_matrices += clock() - t_compose_matrices_start;
}

// compute the matrices M, f, so that M*qddot = f;
// compute derivatives dM_dq, K = df_dq, D = df_dqdot
void Simulation::computeMatrices(MatrixX& M, VectorX& fr, JacobianMatrixVector& dM_dq, MatrixX& K, MatrixX& D) {

    auto t_compute_matrices_start = clock();

    _robot->update();

    VectorX qdot = get_qdot();

    _robot->computeJointJacobianWithDerivative(_J, _Jdot, _dJ_dq, _dJdot_dq);
    
    _robot->computeMaximalMassMatrix(_Mm);

    auto t_compute_force_start = clock();

    _robot->computeForceWithDerivative(_fm, _fr, _Km, _Dm, _Kr, _Dr);
    
    _time_report._time_compute_force += clock() - t_compute_force_start;

    _time_report._time_compute_matrices += clock() - t_compute_matrices_start;

    auto t_compose_matrices_start = clock();

    MatrixX JT = _J.transpose();

    MatrixX MmJ(_ndof_m, _ndof_r);
    for (int i = 0;i < MmJ.rows();i++)
        MmJ.row(i).noalias() = _J.row(i) * _Mm(i);

    MatrixX MmJdot(_ndof_m, _ndof_r);
    for (int i = 0;i < MmJdot.rows();i++)
        MmJdot.row(i).noalias() = _Jdot.row(i) * _Mm(i);

    M = JT * MmJ;

    MatrixX JTMm = MmJ.transpose();

    // fqvv = -J'*Mm*Jdot*qdot
    MatrixX JTMmJdot = JTMm * _Jdot;
    VectorX fqvv = -JTMmJdot * qdot;

    fr = JT * _fm + fqvv + _fr;
    
    /********************************* Derivatives ************************************/
    // dM_dq = (dJ_dq)'*Mm*J + J'*Mm*dJ_dq
    for (int k = 0;k < _ndof_r;k++) {
        MatrixX tmp = _dJ_dq(k).transpose() * MmJ;
        dM_dq(k) = tmp + tmp.transpose();
    }
    
    // dphi_dq = outer_product(dJ_dq, qdot)
    // dphi_dq(k) = dJdq(k) * qdot
    for (int k = 0;k < _ndof_r;k++) {
        _dphi_dq.col(k) = _dJ_dq(k) * qdot;
    }

    // Kqvv = dfqvv_dq = -(dJ_dq)'*Mm*Jdot*qdot-J'*Mm*dJdot_dq*qdot
    // Dqvv = dfqvv_dqdot = -J'*Mm*dJdot_dqdot*qdot-J'*Mm*Jdot = -J'*Mm*dJ_dq*qdot-J'*Mm*Jdot (using the fact dJdot_dqdot = dJ_dq)
    MatrixX Kqvv(_ndof_r, _ndof_r);
    MatrixX Dqvv = -JTMmJdot - JTMm * _dphi_dq;
    VectorX MmJdotqdot = MmJdot * qdot;
    for (int k = 0;k < _ndof_r;k++) {
        Kqvv.col(k) = -_dJ_dq(k).transpose() * MmJdotqdot - JTMm * (_dJdot_dq(k) * qdot);
    }

    // Km = d(J'*fm)_dq = dJ'_dq * fm + J'*(Km*J + Dm * dqmdot_dqr), notes eq (3.49)
    // Dm = d(J'*fm)_dqdot = J'*Dm*J
    MatrixX JTDm = JT * _Dm;
    K = Kqvv + _Kr + JT * _Km * _J + JTDm * _dphi_dq;
    for (int k = 0;k < _ndof_r;k++) {
        K.col(k) += _dJ_dq(k).transpose() * _fm;
    }

    D = Dqvv + _Dr + JTDm * _J;

    _time_report._time_compose_matrices += clock() - t_compose_matrices_start;
}

// compute the matrices M, f, so that M*qddot = f;
// compute derivatives dM_dq, K = df_dq, D = df_dqdot
// compute derivatives w.r.t control
void Simulation::computeMatrices(MatrixX& M, VectorX& fr, JacobianMatrixVector& dM_dq, MatrixX& K, MatrixX& D, MatrixX& dfr_du) {

    auto t_compute_matrices_start = clock();

    _robot->update();

    VectorX qdot = get_qdot();

    _robot->computeJointJacobianWithDerivative(_J, _Jdot, _dJ_dq, _dJdot_dq);

    _robot->computeMaximalMassMatrix(_Mm);
    
    auto t_compute_force_start = clock();

    _robot->computeForceWithDerivative(_fm, _fr, _Km, _Dm, _Kr, _Dr);

    _time_report._time_compute_force += clock() - t_compute_force_start;

    _time_report._time_compute_matrices += clock() - t_compute_matrices_start;

    auto t_compose_matrices_start = clock();

    MatrixX JT = _J.transpose();

    MatrixX MmJ(_ndof_m, _ndof_r);
    for (int i = 0;i < MmJ.rows();i++)
        MmJ.row(i).noalias() = _J.row(i) * _Mm(i);

    MatrixX MmJdot(_ndof_m, _ndof_r);
    for (int i = 0;i < MmJdot.rows();i++)
        MmJdot.row(i).noalias() = _Jdot.row(i) * _Mm(i);

    M = JT * MmJ;

    MatrixX JTMm = MmJ.transpose(); // TODO: need to check whether Mm is symmetric

    // fqvv = -J'*Mm*Jdot*qdot
    MatrixX JTMmJdot = JTMm * _Jdot;
    VectorX fqvv = -JTMmJdot * qdot;

    fr = JT * _fm + fqvv + _fr;
    
    /********************************* Derivatives ************************************/
    // dM_dq = (dJ_dq)'*Mm*J + J'*Mm*dJ_dq
    for (int k = 0;k < _ndof_r;k++) {
        MatrixX tmp = _dJ_dq(k).transpose() * MmJ;
        dM_dq(k) = tmp + tmp.transpose();
    }

    // dphi_dq = outer_product(dJ_dq, qdot)
    // dphi_dq(k) = dJdq(k) * qdot
    for (int k = 0;k < _ndof_r;k++) {
        _dphi_dq.col(k) = _dJ_dq(k) * qdot;
    }

    // Kqvv = dfqvv_dq = -(dJ_dq)'*Mm*Jdot*qdot-J'*Mm*dJdot_dq*qdot
    // Dqvv = dfqvv_dqdot = -J'*Mm*dJdot_dqdot*qdot-J'*Mm*Jdot = -J'*Mm*dJ_dq*qdot-J'*Mm*Jdot (using the fact dJdot_dqdot = dJ_dq)
    MatrixX Kqvv(_ndof_r, _ndof_r);
    MatrixX Dqvv = -JTMmJdot - JTMm * _dphi_dq;
    VectorX MmJdotqdot = MmJdot * qdot;
    for (int k = 0;k < _ndof_r;k++) {
        Kqvv.col(k) = -_dJ_dq(k).transpose() * MmJdotqdot - JTMm * (_dJdot_dq(k) * qdot);
    }

    // Km = d(J'*fm)_dq = dJ'_dq * fm + J'*(Km*J + Dm * dqmdot_dqr), notes eq (3.49)
    // Dm = d(J'*fm)_dqdot = J'*Dm*J
    MatrixX JTDm = JT * _Dm;
    K = Kqvv + _Kr + JT * _Km * _J + JTDm * _dphi_dq;
    for (int k = 0;k < _ndof_r;k++) {
        K.col(k) += _dJ_dq(k).transpose() * _fm;
    }

    D = Dqvv + _Dr + JTDm * _J;

    // df_du = JT * dfm_du + dfr_du
    _robot->compute_dfdu(_dfm_du, dfr_du);
    dfr_du += JT * _dfm_du;

    _time_report._time_compose_matrices += clock() - t_compose_matrices_start;
}

// compute the matrices M, f, so that M*qddot = f;
// compute derivatives dM_dq, K = df_dq, D = df_dqdot
// compute derivatives w.r.t control
// compute derivatives w.r.t design parameters
void Simulation::computeMatrices(
        MatrixX& M, VectorX& fr,
        JacobianMatrixVector& dM_dq, MatrixX& K, MatrixX& D,
        MatrixX& dfr_du,
        JacobianMatrixVector& dM_dp, MatrixX& dfr_dp) {

    auto t_compute_matrices_start = clock();

    _robot->update();

    VectorX qdot = get_qdot();

    auto t_compute_dJ_start = clock();

    SparseJacobianMatrixVector dJ_dp1, dJ_dp2, dJdot_dp1, dJdot_dp2;
    _robot->computeJointJacobianWithDerivative(_J, _Jdot, _dJ_dq, _dJdot_dq, dJ_dp1, dJ_dp2, dJdot_dp1, dJdot_dp2);

    _time_report._time_compute_dJ += clock() - t_compute_dJ_start;

    _robot->computeMaximalMassMatrix(_Mm);
    
    auto t_compute_df_start = clock();

    MatrixX dfm_dp;
    _robot->computeForceWithDerivative(_fm, _fr, _Km, _Dm, _Kr, _Dr, dfm_dp, dfr_dp);

    _time_report._time_compute_df += clock() - t_compute_df_start;

    // _time_report._time_compute_matrices += clock() - t_compute_matrices_start;

    auto t_compose_matrices_start = clock();

    MatrixX JT = _J.transpose();

    MatrixX MmJ(_ndof_m, _ndof_r);
    for (int i = 0;i < MmJ.rows();i++)
        MmJ.row(i).noalias() = _J.row(i) * _Mm(i);

    MatrixX MmJdot(_ndof_m, _ndof_r);
    for (int i = 0;i < MmJdot.rows();i++)
        MmJdot.row(i).noalias() = _Jdot.row(i) * _Mm(i);

    M = JT * MmJ;

    MatrixX JTMm = MmJ.transpose();

    // fqvv = -J'*Mm*Jdot*qdot
    MatrixX JTMmJdot = JTMm * _Jdot;
    VectorX Jdotqdot = _Jdot * qdot;
    VectorX fqvv = -JTMmJdot * qdot;

    fr = JT * _fm + fqvv + _fr;
    
    /********************************* Derivatives ************************************/
    // dM_dq = (dJ_dq)'*Mm*J + J'*Mm*dJ_dq
    for (int k = 0;k < _ndof_r;k++) {
        MatrixX tmp = _dJ_dq(k).transpose() * MmJ;
        dM_dq(k) = tmp + tmp.transpose();
    }

    // dphi_dq = outer_product(dJ_dq, qdot)
    // dphi_dq(k) = dJdq(k) * qdot
    for (int k = 0;k < _ndof_r;k++) {
        _dphi_dq.col(k) = _dJ_dq(k) * qdot;
    }

    // Kqvv = dfqvv_dq = -(dJ_dq)'*Mm*Jdot*qdot-J'*Mm*dJdot_dq*qdot
    // Dqvv = dfqvv_dqdot = -J'*Mm*dJdot_dqdot*qdot-J'*Mm*Jdot = -J'*Mm*dJ_dq*qdot-J'*Mm*Jdot (using the fact dJdot_dqdot = dJ_dq)
    MatrixX Kqvv(_ndof_r, _ndof_r);
    MatrixX Dqvv = -JTMmJdot - JTMm * _dphi_dq;
    VectorX MmJdotqdot = MmJdot * qdot;
    for (int k = 0;k < _ndof_r;k++) {
        Kqvv.col(k) = -_dJ_dq(k).transpose() * MmJdotqdot - JTMm * (_dJdot_dq(k) * qdot);
    }

    // Km = d(J'*fm)_dq = dJ'_dq * fm + J'*(Km*J + Dm * dqmdot_dqr), notes eq (3.49)
    // Dm = d(J'*fm)_dqdot = J'*Dm*J
    MatrixX JTDm = JT * _Dm;
    K = Kqvv + _Kr + JT * _Km * _J + JTDm * _dphi_dq;
    for (int k = 0;k < _ndof_r;k++) {
        K.col(k) += _dJ_dq(k).transpose() * _fm;
    }

    D = Dqvv + _Dr + JTDm * _J;

    // df_du = JT * dfm_du + dfr_du
    _robot->compute_dfdu(_dfm_du, dfr_du);
    dfr_du += JT * _dfm_du;

    // design derivatives

    auto t_dM_dp_start = clock();

    // dM_dp
    auto t_dM_dp1_start = clock();
    // design params 1
    for (int i = 0;i < _ndof_p1;i++) {
        MatrixX tmp = dJ_dp1(i).transpose() * MmJ;
        dM_dp(i) = tmp + tmp.transpose();
    }
    _time_report._time_dM_dp1 += clock() - t_dM_dp1_start;
    auto t_dM_dp2_start = clock();
    // design params 2
    for (int i = 0;i < _ndof_p2;i++) {
        MatrixX tmp = dJ_dp2(i).transpose() * MmJ;
        dM_dp(_ndof_p1 + i) = tmp + tmp.transpose();
    }
    _time_report._time_dM_dp2 += clock() - t_dM_dp2_start;
    auto t_dM_dp4_start = clock();
    // design params 4
    int p4_offset = _ndof_p1 + _ndof_p2 + _ndof_p3;
    for (auto body : _robot->_bodies) {
        if (body->_design_params_4._active) {
            int m_id = body->_index[0];
            int p_id = body->_design_params_4._param_index[0];
            // mass
            dM_dp(p4_offset + p_id) = JT.middleCols(m_id + 3, 3) * _J.middleRows(m_id + 3, 3);
            // inertia
            for (int k = 0;k < 3;k++) {
                dM_dp(p4_offset + p_id + 1 + k) = JT.col(m_id + k) * _J.row(m_id + k);
            }
        }
    }
    _time_report._time_dM_dp4 += clock() - t_dM_dp4_start;
    _time_report._time_dM_dp += clock() - t_dM_dp_start;

    auto t_df_dp_start = clock();

    // fqvv component
    // design params 1
    for (int i = 0;i < _ndof_p1;i++) {
        dfr_dp.col(i) -= dJ_dp1(i).transpose() * MmJdotqdot + JTMm * dJdot_dp1(i) * qdot;
    }
    // design params 2
    for (int i = 0;i < _ndof_p2;i++) {
        dfr_dp.col(i + _ndof_p1) -= dJ_dp2(i).transpose() * MmJdotqdot + JTMm * dJdot_dp2(i) * qdot;
    }
    // design params 4
    for (auto body : _robot->_bodies) {
        if (body->_design_params_4._active) {
            int m_id = body->_index[0];
            int p_id = body->_design_params_4._param_index[0];
            // mass
            dfr_dp.col(p4_offset + p_id) -= JT.middleCols(m_id + 3, 3) * Jdotqdot.segment(m_id + 3, 3);
            // inertia
            for (int k = 0;k < 3;k++) {
                dfr_dp.col(p4_offset + p_id + 1 + k) -= JT.col(m_id + k) * Jdotqdot(m_id + k);
            }
        }
    }

    // fm component: dJdp * fm + JT * dfm_dp
    // dJdp * fm
    // design params 1
    for (int i = 0;i < _ndof_p1;i++) {
        dfr_dp.col(i) += dJ_dp1(i).transpose() * _fm;
    }
    // design params 2
    for (int i = 0;i < _ndof_p2;i++) {
        dfr_dp.col(i + _ndof_p1) += dJ_dp2(i).transpose() * _fm;
    }
    // JT * dfm_dp
    for (int i = 0;i < _ndof_p;i++) {
        dfr_dp.col(i) += JT * dfm_dp.col(i);
    }
    
    _time_report._time_df_dp += clock() - t_df_dp_start;

    _time_report._time_compose_matrices += clock() - t_compose_matrices_start;
}


// derivatives w.r.t. previous states, now only for fr
// e.g. compute the df_dqprev and df_dqdotprev introduced by positional control
void Simulation::computeExtraDerivative(MatrixX& dfr_dqprev, MatrixX& dfr_dqdotprev) {
    _robot->computeExtraDerivative(dfr_dqprev, dfr_dqdotprev);
}

// compute M(q)x + y*f(q) using jacobian product
/*
x \in R^{ndof_r}
y is scalar
g = M(q) * x + y * f(q)
*/
void Simulation::computeMatricesProduct(const VectorX& x, const dtype y, VectorX& g) { 
    _robot->update();

    /**** M = J'*Mm*J -> Mx = J'*Mm*Jx ****/
    
    // Jx = J * x
    VectorX Jx;
    _robot->computeJointJacobianProduct(x, Jx);
    
    // MmJx = Mm * Jx
    VectorX Mm;
    _robot->computeMaximalMassMatrix(Mm);
    VectorX MmJx = Mm.cwiseProduct(Jx);

    // Mx = J' * MmJx
    VectorX Mx;
    _robot->computeJointJacobianTransposeProduct(MmJx, Mx);

    /**** f = J' * (fm - Mm * Jdot * qdot) + fr ****/
    
    // Jdotqdot = Jdot * qdot
    VectorX qdot = get_qdot();
    VectorX Jdotqdot;
    _robot->computeJointJacobianDotProduct(qdot, Jdotqdot);
    
    // MmJdotqdot = Mm * Jdotqdot
    VectorX MmJdotqdot = Mm.cwiseProduct(Jdotqdot);
    
    // f = fm - MmJdotqdot
    VectorX fm, fr;
    _robot->computeForce(fm, fr);
    VectorX f = fm - MmJdotqdot;
    
    // JTf = J' * f
    VectorX JTf;
    _robot->computeJointJacobianTransposeProduct(f, JTf);

    // f = JTf + fr
    f = JTf + fr;

    g = Mx + y * f;
}

/* 
g = M(q) * x + y * f(q)
H_sub.col(k) = dM_dq(k) * x
*/
void Simulation::computeMatricesProduct(const VectorX& x, const dtype y, VectorX& g, 
                                            MatrixX& H_sub, MatrixX& M, MatrixX& K, MatrixX& D) {
    
    _robot->update();

    VectorX qdot = get_qdot();
    
    /**** M = J'*Mm*J -> Mx = J'*Mm*Jx ****/
    
    // Jx = J * x
    VectorX Jx;
    _robot->computeJointJacobianProduct(x, Jx);
    
    // MmJx = Mm * Jx
    VectorX Mm;
    _robot->computeMaximalMassMatrix(Mm);
    VectorX MmJx = Mm.cwiseProduct(Jx);

    // Mx = J' * MmJx
    VectorX Mx;
    _robot->computeJointJacobianTransposeProduct(MmJx, Mx);

    VectorX fm, fr;
    MatrixX Km, Dm, Kr, Dr;
    _robot->computeForceWithDerivative(fm, fr, Km, Dm, Kr, Dr);

    /**** f = J' * (fm - Mm * Jdot * qdot) + fr ****/
    
    // Jdotqdot = Jdot * qdot
    VectorX Jdotqdot;
    _robot->computeJointJacobianDotProduct(qdot, Jdotqdot);
    
    // MmJdotqdot = Mm * Jdotqdot
    VectorX MmJdotqdot = Mm.cwiseProduct(Jdotqdot);
    
    // f = fm - MmJdotqdot
    VectorX f = fm - MmJdotqdot;
    
    // JTf = J' * f
    VectorX JTf;
    _robot->computeJointJacobianTransposeProduct(f, JTf);

    // f = JTf + fr
    f = JTf + fr;

    g = Mx + y * f;

    /********************************* Matrices and Derivatives ************************************/
    
    // Kqvv = dfqvv_dq = -(dJ_dq)'*Mm*Jdot*qdot-J'*Mm*dJdot_dq*qdot
    // Dqvv = dfqvv_dqdot = -J'*Mm*dJdot_dqdot*qdot-J'*Mm*Jdot = -J'*Mm*dJ_dq*qdot-J'*Mm*Jdot (using the fact dJdot_dqdot = dJ_dq)
    auto t0 = clock();
    // need dJdq_qdot: dJdq_qdot.col(k) = dJ_dq(k) * qdot
    MatrixX dJdq_qdot;
    _robot->computeJointJacobianDerivativeProduct(qdot, dJdq_qdot);
    // need dJTdq_MmJdotqdot: dJTdq_MmJdotqdot.col(k) = dJT_dq(k) * MmJdotqdot
    MatrixX dJTdq_MmJdotqdot;
    _robot->computeJointJacobianTransposeDerivativeProduct(MmJdotqdot, dJTdq_MmJdotqdot);
    // need dJdotdq_qdot: dJdotdq_qdot.col(k) = dJdot_dq(k) * qdot
    MatrixX dJdotdq_qdot;
    _robot->computeJointJacobianDotDerivativeProduct(qdot, dJdotdq_qdot);
    // need dJTdq_fm: dfTdq_fm.col(k) = dJT_dq(k) * fm
    MatrixX dJTdq_fm;
    _robot->computeJointJacobianTransposeDerivativeProduct(fm, dJTdq_fm);
    auto t1 = clock();
    _time_report._time_compute_dJ += t1 - t0;

    MatrixX J, Jdot;
    _robot->computeJointJacobian(J, Jdot);

    MatrixX JT = J.transpose();
    
    MatrixX MmJ(_ndof_m, _ndof_r);
    for (int i = 0;i < MmJ.rows();i++)
        MmJ.row(i).noalias() = J.row(i) * Mm(i);

    MatrixX MmJdot(_ndof_m, _ndof_r);
    for (int i = 0;i < MmJdot.rows();i++)
        MmJdot.row(i).noalias() = Jdot.row(i) * Mm(i);

    M = JT * MmJ;

    MatrixX JTMm = MmJ.transpose();
    MatrixX JTMmJdot = JTMm * Jdot;
    MatrixX Kqvv = -dJTdq_MmJdotqdot;
    MatrixX Dqvv = -JTMmJdot - JTMm * dJdq_qdot;
    
    for (int k = 0;k < _ndof_r;k++) {
        VectorX tmp_x = Mm.cwiseProduct(dJdotdq_qdot.col(k));
        VectorX tmp;
        _robot->computeJointJacobianTransposeProduct(tmp_x, tmp);
        Kqvv.col(k) -= tmp;
    }

    // Km = d(J'*fm)_dq = dJ'_dq * fm + J'*(Km*J + Dm * dqmdot_dqr), notes eq (3.49)
    // Dm = d(J'*fm)_dqdot = J'*Dm*J
    MatrixX JTDm = JT * Dm;
    K = Kqvv + Kr + JT * Km * J + JTDm * dJdq_qdot + dJTdq_fm;

    D = Dqvv + Dr + JTDm * J;

    // H_sub.col(k) = dM_dq(k) * x
    // dM_dq = (dJ_dq)'*Mm*J + J'*Mm*dJ_dq
    t0 = clock();
    _robot->computeJointJacobianTransposeDerivativeProduct(MmJx, H_sub);
    MatrixX dJdq_x;
    _robot->computeJointJacobianDerivativeProduct(x, dJdq_x);
    H_sub += JTMm * dJdq_x;
    t1 = clock();
    _time_report._time_compute_dJ += t1 - t0;
}

void Simulation::computeVariablesWithDerivative(VectorX& variables, MatrixX& dvar_dq) {
    _robot->computeVariablesWithDerivative(variables, dvar_dq);
}

void Simulation::computeVariablesWithDerivative(VectorX& variables, MatrixX& dvar_dq, MatrixX& dvar_dp) {
    _robot->computeVariablesWithDerivative(variables, dvar_dq, dvar_dp);
}

void Simulation::computeTactileWithDerivatives(VectorX& tactile_force, MatrixX& dtactile_dq, MatrixX& dtactile_dqdot) {
    // assume the latest J and Jdot and _dphi_dq has been computed. TODO: check this
    dtactile_dq = MatrixX::Zero(_ndof_tactile, _ndof_r);
    dtactile_dqdot = MatrixX::Zero(_ndof_tactile, _ndof_r);
    
    MatrixX dtactile_dqm, dtactile_dphi;
    std::vector<pair<Body*, Body*>> contact_bodies;
    _robot->computeTactileWithDerivatives(tactile_force, dtactile_dqm, dtactile_dphi, contact_bodies);
    
    assert(tactile_force.size() == _ndof_tactile);

    for (int i = 0;i < contact_bodies.size();++i) 
        if (contact_bodies[i].second != nullptr) {
            // contribution from body 1
            dtactile_dq.middleRows(i * 3, 3) += 
                dtactile_dqm.block(i * 3, 0, 3, 6) * _J.middleRows(contact_bodies[i].first->_index[0], 6) +
                dtactile_dphi.block(i * 3, 0, 3, 6) * _dphi_dq.middleRows(contact_bodies[i].first->_index[0], 6);
            dtactile_dqdot.middleRows(i * 3, 3) +=
                dtactile_dphi.block(i * 3, 0, 3, 6) * _J.middleRows(contact_bodies[i].first->_index[0], 6);

            // contribution from body 2
            dtactile_dq.middleRows(i * 3, 3) += 
                dtactile_dqm.block(i * 3, 6, 3, 6) * _J.middleRows(contact_bodies[i].second->_index[0], 6) +
                dtactile_dphi.block(i * 3, 6, 3, 6) * _dphi_dq.middleRows(contact_bodies[i].second->_index[0], 6);
            dtactile_dqdot.middleRows(i * 3, 3) +=
                dtactile_dphi.block(i * 3, 6, 3, 6) * _J.middleRows(contact_bodies[i].second->_index[0], 6);
        }
}

void Simulation::test_derivatives_runtime() {
    _robot->test_derivatives_runtime();

    // test dM_dq, K, D
    _robot->update();

    auto q = get_q();
    auto qdot = get_qdot();

    MatrixX M = MatrixX::Zero(_ndof_r, _ndof_r);
    VectorX fr = VectorX::Zero(_ndof_r);
    JacobianMatrixVector dM_dq(_ndof_r, _ndof_r, _ndof_r);
    MatrixX K = MatrixX::Zero(_ndof_r, _ndof_r), D = MatrixX::Zero(_ndof_r, _ndof_r);
    computeMatrices(M, fr, dM_dq, K, D);

    MatrixX dtactile_dq = MatrixX::Zero(_ndof_tactile, _ndof_r), dtactile_dqdot = MatrixX::Zero(_ndof_tactile, _ndof_r);
    VectorX tactile_force;

    computeTactileWithDerivatives(tactile_force, dtactile_dq, dtactile_dqdot);

    // printf("**************************** Simulation ****************************\n");
    dtype h = 1e-7;
    for (int ii = 0;ii < 1;ii++) {
        // printf("---------------------- eps = %.9lf ----------------------------\n", h);
        JacobianMatrixVector dM_dq_fd(_ndof_r, _ndof_r, _ndof_r);
        MatrixX K_fd = MatrixX::Zero(_ndof_r, _ndof_r);
        for (int k = 0;k < _ndof_r;k++) {
            auto q_pos = q;
            q_pos(k) += h;
            set_q(q_pos);

            _robot->update();

            MatrixX M_pos = MatrixX::Zero(_ndof_r, _ndof_r);
            VectorX fr_pos = VectorX::Zero(_ndof_r);
            computeMatrices(M_pos, fr_pos);

            dM_dq_fd(k) = (M_pos - M) / h;
            K_fd.col(k) = (fr_pos - fr) / h;
        }

        print_error("Simulation: dM_dq", dM_dq, dM_dq_fd);
        print_error("Simulation: K", K, K_fd);

        set_q(q);
        _robot->update();

        MatrixX D_fd = MatrixX::Zero(_ndof_r, _ndof_r);
        for (int k = 0;k < _ndof_r;k++) {
            auto qdot_pos = qdot;
            qdot_pos(k) += h;
            set_qdot(qdot_pos);

            _robot->update();

            MatrixX M_pos = MatrixX::Zero(_ndof_r, _ndof_r);;
            VectorX fr_pos = VectorX::Zero(_ndof_r);;
            computeMatrices(M_pos, fr_pos);

            D_fd.col(k) = (fr_pos - fr) / h;
        }

        print_error("Simulation: D", D, D_fd);

        set_qdot(qdot);
        _robot->update();

        // dtactile_dq
        MatrixX dtactile_dq_fd(_ndof_tactile, _ndof_r);
        for (int k = 0;k < _ndof_r;k++) {
            auto q_pos = q;
            q_pos(k) += h;
            set_q(q_pos);

            _robot->update();

            VectorX tactile_force_pos = VectorX::Zero(_ndof_tactile);

            MatrixX tmp1 = MatrixX::Zero(_ndof_tactile, _ndof_r);
            MatrixX tmp2 = MatrixX::Zero(_ndof_tactile, _ndof_r);
            computeTactileWithDerivatives(tactile_force_pos, tmp1, tmp2);

            dtactile_dq_fd.col(k) = (tactile_force_pos - tactile_force) / h;
        }

        // char str[100];
        // print_error_full("Simulation: dtactile_dq", dtactile_dq, dtactile_dq_fd);
        print_error("Simulation: dtactile_dq", dtactile_dq, dtactile_dq_fd);
        // std::cerr << dtactile_dq.norm() << std::endl;

        set_q(q);
        _robot->update();

        // dtactile_dqdot
        MatrixX dtactile_dqdot_fd(_ndof_tactile, _ndof_r);
        for (int k = 0;k < _ndof_r;k++) {
            auto qdot_pos = qdot;
            qdot_pos(k) += h;
            set_qdot(qdot_pos);

            _robot->update();

            VectorX tactile_force_pos = VectorX::Zero(_ndof_tactile);

            MatrixX tmp1 = MatrixX::Zero(_ndof_tactile, _ndof_r);
            MatrixX tmp2 = MatrixX::Zero(_ndof_tactile, _ndof_r);
            computeTactileWithDerivatives(tactile_force_pos, tmp1, tmp2);

            dtactile_dqdot_fd.col(k) = (tactile_force_pos - tactile_force) / h;
        }

        // char str[100];
        // print_error_full("Simulation: dtactile_dqdot", dtactile_dqdot, dtactile_dqdot_fd);
        print_error("Simulation: dtactile_dqdot", dtactile_dqdot, dtactile_dqdot_fd);

        set_qdot(qdot);
        _robot->update();

        h /= 10.;
    }

    test_design_derivatives_runtime();
}

void Simulation::test_design_derivatives_runtime() {
    dtype eps = 1e-8;

    VectorX design_params = get_design_params();
    _robot->update(true);

    MatrixX M;
    VectorX fr;
    JacobianMatrixVector dM_dq;
    MatrixX K, D;
    MatrixX dfr_du;
    computeMatrices(M, fr, dM_dq, K, D, dfr_du, _dM_dp, _dfr_dp);

    if (_ndof_p > 0) {
        JacobianMatrixVector dM_dp_fd(_ndof_r, _ndof_r, _ndof_p);
        MatrixX dfr_dp_fd = MatrixX::Zero(_ndof_r, _ndof_p);
        for (int i = 0;i < _ndof_p;i++) {
            VectorX design_params_pos = design_params;
            design_params_pos(i) += eps;
            set_design_params(design_params_pos);
            _robot->update(false);
            MatrixX M_pos;
            VectorX fr_pos;
            computeMatrices(M_pos, fr_pos);
            dM_dp_fd(i) = (M_pos - M) / eps;
            dfr_dp_fd.col(i) = (fr_pos - fr) / eps;
        }
        print_error("Simulation: dM_dp", _dM_dp, dM_dp_fd);
        print_error("Simulation: dfr_dp", _dfr_dp, dfr_dp_fd);
    }

    set_design_params(design_params);
    _robot->update(true);
}

void Simulation::reset(bool backward_flag, bool backward_design_params_flag) {
    set_state(_q_init, _qdot_init);
    update_robot();

    _robot->update_actuator_states(_q_init, _qdot_init);
    
    _q_his.clear();
    _qdot_his.clear();
    _q_his.push_back(_q_init);
    _qdot_his.push_back(_qdot_init);

    _virtual_object_data_his.clear();
    _virtual_object_data_his.push_back(_robot->get_virtual_object_data());

    _backward_flag = backward_flag;
    _backward_design_params_flag = backward_design_params_flag;
    if (_ndof_p == 0) {
        _backward_design_params_flag = false;
    }

    if (!_backward_flag && _backward_design_params_flag) {
        std::cerr << "[Error] _backward_design_params_flag is True while _backward_flag is False" << std::endl;
        throw "error";
    }
    
    _backward_info.clear();

    if (backward_flag) {
        _backward_info._q_his.push_back(_q_init);
        _backward_info._qdot_his.push_back(_qdot_init);
    }

    _time_report.reset();
    _robot->reset_time_report();
}

// manipulate history cache for backward info and backward results
void Simulation::clearBackwardCache() {
    _backward_info_his.clear();
    _backward_results_his.clear();
}

void Simulation::saveBackwardCache() {
    _backward_info_his.push_back(_backward_info);
    _backward_results_his.push_back(_backward_results);
}

void Simulation::popBackwardCache() {
    _backward_info = _backward_info_his[_backward_info_his.size() - 1];
    _backward_info_his.pop_back();
    _backward_results = _backward_results_his[_backward_results_his.size() - 1];
    _backward_results_his.pop_back();
}

int Simulation::backwardCacheSize() {
    return _backward_info_his.size();
}

void Simulation::forward(int num_steps, bool verbose, bool test_derivatives, bool save_last_frame_var_only) {
    
    _verbose = verbose;

    if (_q_his.size() == 0) {
        std::cerr << "[Error] Please call simulation.reset() before simulation.forward()." << std::endl;
        throw "error";
    }

    for (int i = 0;i < num_steps;i++) {
        VectorX q = get_q();
        VectorX qdot = get_qdot();

        // if (verbose) {
        //     std::cerr << "q = " << q.transpose() << ", qdot = " << qdot.transpose() << std::endl;
        // }

        assert((q - _q_his[_q_his.size() - 1]).norm() < 1e-7 && (qdot - _qdot_his[_qdot_his.size() - 1]).norm() < 1e-7);

        VectorX q_next, qdot_next;
        if (_options->_integrator == "BDF1") {
            integration_BDF1(q, qdot, _options->_h, q_next, qdot_next);
        } else if (_options->_integrator == "BDF2") {
            if (_q_his.size() == 1) {
                integration_SDIRK2(q, qdot, _options->_h, q_next, qdot_next);
            } else{
                VectorX q_prev = _q_his[_q_his.size() - 2];
                VectorX qdot_prev = _qdot_his[_qdot_his.size() - 2];
                integration_BDF2(q_prev, qdot_prev, q, qdot, _options->_h, q_next, qdot_next);
            }
        } else if (_options->_integrator == "SDIRK2") {
            integration_SDIRK2(q, qdot, _options->_h, q_next, qdot_next);
        } else {
            std::cerr << "[Error] Integrator " << _options->_integrator << " has not been implemented." << std::endl;
            throw "error";
        }

        set_state(q_next, qdot_next);
        // reparam();
        q_next = get_q();
        qdot_next = get_qdot();
        
        update_robot(_backward_flag);

        // update actuator's dofs
        _robot->update_actuator_states(q_next, qdot_next);

        _q_his.push_back(q_next);
        _qdot_his.push_back(qdot_next);

        _virtual_object_data_his.push_back(_robot->get_virtual_object_data());

        int t_save_backward_start = clock();
        if (_backward_flag) {
            _backward_info._current_backward_step += 1;
            _backward_info._q_his.push_back(q_next);
            _backward_info._qdot_his.push_back(qdot_next);
            if (!save_last_frame_var_only || (i == num_steps - 1)) {
                if (_backward_design_params_flag) {
                    VectorX variables;
                    MatrixX dvar_dq, dvar_dp;
                    computeVariablesWithDerivative(variables, dvar_dq, dvar_dp);
                    _backward_info._dvar_dq.push_back(dvar_dq);
                    _backward_info._dvar_dp.push_back(dvar_dp);
                } else {
                    VectorX variables;
                    MatrixX dvar_dq;
                    computeVariablesWithDerivative(variables, dvar_dq);
                    _backward_info._dvar_dq.push_back(dvar_dq);
                }
                // TODO: design derivatives for tactile
                if (_ndof_tactile > 0) {
                    VectorX tactile_force;
                    MatrixX dtactile_dq, dtactile_dqdot;
                    computeTactileWithDerivatives(tactile_force, dtactile_dq, dtactile_dqdot);
                    _backward_info._dtactile_dq.push_back(dtactile_dq);
                    _backward_info._dtactile_dqdot.push_back(dtactile_dqdot);
                }
            } else {
                _backward_info._dvar_dq.push_back(MatrixX::Zero(_ndof_var, _ndof_r));
                _backward_info._dtactile_dq.push_back(MatrixX::Zero(_ndof_tactile, _ndof_r));
                _backward_info._dtactile_dqdot.push_back(MatrixX::Zero(_ndof_tactile, _ndof_r));
            }
        }

        if (test_derivatives) {
            test_derivatives_runtime();
        }

        _time_report._time_save_backward += clock() - t_save_backward_start;
    }
}

void Simulation::newton(VectorX& x,
                        Func func, 
                        Func_With_Derivatives func_with_derivatives) {
    
    auto t_newton_start = clock();

    if (_ndof_r == 0)
        return;

    dtype tol = _solver_options->_tol;
    int MaxIter_Newton = max(20 * _ndof_r, _solver_options->_MaxIter_Newton);
    int MaxIter_LS = _solver_options->_MaxIter_LS;

    // int MaxIter_LS_Fail_Strike = 1;
    int MaxIter_LS_Fail_Strike = 10;

    bool success_newton = false;
    dtype g_last;
    int fail_strike = 0;
    std::vector<dtype> g_his;
    // std::cerr << "newton's solver" << std::endl;
    for (int iter_newton = 0;iter_newton < MaxIter_Newton;iter_newton ++) {
        VectorX g;
        MatrixX H;
        (this->*func_with_derivatives)(x, g, H, false);

        // new_x = -inv(H) * g + x
        // VectorX dx = -H.inverse() * g;
        VectorX dx = H.partialPivLu().solve(-g);

        // line search new_x = x + alpha * dx
        VectorX g_new;
        dtype gnorm = g.norm();
        dtype alpha = 1.;
        bool success_ls = false;
        // for (int trial = 0;trial < MaxIter_LS;trial ++, alpha *= 0.95) {
        for (int trial = 0;trial < MaxIter_LS;trial ++, alpha *= 0.5) {
            auto t_eval_start = clock();

            (this->*func)(x + alpha * dx, g_new);

            if (g_new.norm() < gnorm) {
                success_ls = true;
                break;
            }
        }

        // std::cerr << "g_new = " << g_new.norm() << ", g = " << gnorm << std::endl;

        if (success_ls) {
            fail_strike = 0;
        } else {
            fail_strike += 1;
            if (fail_strike >= MaxIter_LS_Fail_Strike)
                break;
        }

        x = x + alpha * dx;

        if (g_new.norm() < tol) {
            success_newton = true;
            break;
        }

        g_last = g_new.norm();
        g_his.push_back(g_last);
    }

    if (!success_newton && (g_last > 1e-5 || isnan(g_last))) {
        if (_verbose) {
            std::cerr << "Newton method did not converge. g = " << g_last << std::endl;
        }
    }

    _time_report._time_solver += clock() - t_newton_start;
}

void Simulation::evaluate_g_BDF1(const VectorX& q1, VectorX& g) {
    VectorX qdot1 = (q1 - _q0) / _h;

    set_state(q1, qdot1);
    update_robot();

    computeMatrices(_M, _fr);
    g = _M * (q1 - _q0 - _h * _qdot0) - _h * _h * _fr;
}

void Simulation::evaluate_g_with_derivatives_BDF1(const VectorX& q1, VectorX& g, MatrixX& H, bool save_backward_info) {
    VectorX qdot1 = (q1 - _q0) / _h;

    set_state(q1, qdot1);
    update_robot(save_backward_info && _backward_design_params_flag);

    if (!save_backward_info) {
        computeMatrices(_M, _fr, _dM_dq, _K, _D);

        VectorX dq_tmp = q1 - _q0 - _h * _qdot0;
        g = _M * dq_tmp - _h * _h * _fr;
        H = _M - _h * _h * _K - _h * _D;
        for (int k = 0;k < _ndof_r;k++) {
            H.col(k) += _dM_dq(k) * dq_tmp;
        }
    } else {
        if (_backward_design_params_flag) {
            computeMatrices(_M, _fr, _dM_dq, _K, _D, _dfr_du, _dM_dp, _dfr_dp);
            computeExtraDerivative(_dfr_dqprev, _dfr_dqdotprev);

            VectorX dq_tmp = q1 - _q0 - _h * _qdot0;
            g = _M * dq_tmp - _h * _h * _fr;
            H = _M - _h * _h * _K - _h * _D;
            for (int k = 0;k < _ndof_r;k++) {
                H.col(k) += _dM_dq(k) * dq_tmp;
            }
            MatrixX dg_dp = -_h * _h * _dfr_dp;
            // for (int k = 0;k < _ndof_p;k++) {
            //     dg_dp.col(k) += dM_dp(k) * dq_tmp;
            // }
            for (int k = 0;k < _ndof_p1 + _ndof_p2;k++) {
                dg_dp.col(k) += _dM_dp(k) * dq_tmp;
            }
            for (int k = _ndof_p1 + _ndof_p2 + _ndof_p3;k < _ndof_p1 + _ndof_p2 + _ndof_p3 + _ndof_p4;k++) {
                dg_dp.col(k) += _dM_dp(k) * dq_tmp;
            }

            // save backward data
            _backward_info._M.push_back(_M);
            _backward_info._D.push_back(_D);
            _backward_info._dfr_dqprev.push_back(_dfr_dqprev);
            _backward_info._dfr_dqdotprev.push_back(_dfr_dqdotprev);
            _backward_info._dg_dp.push_back(dg_dp);
            _backward_info._dg_du.push_back(-_h * _h * _dfr_du);
        } else {
            computeMatrices(_M, _fr, _dM_dq, _K, _D, _dfr_du);
            computeExtraDerivative(_dfr_dqprev, _dfr_dqdotprev);

            VectorX dq_tmp = q1 - _q0 - _h * _qdot0;
            g = _M * dq_tmp - _h * _h * _fr;
            H = _M - _h * _h * _K - _h * _D;
            for (int k = 0;k < _ndof_r;k++) {
                H.col(k) += _dM_dq(k) * dq_tmp;
            }

            // // DEBUG test H
            // dtype eps = 1e-3;
            // for (int ii = 0;ii < 7;++ii) {
            //     printf("---------------------- eps = %.9lf ----------------------------\n", eps);

            //     MatrixX H_fd = MatrixX::Zero(_ndof_r, _ndof_r);
            //     for (int k = 0;k < _ndof_r;++k) {
            //         auto q_pos = q1;
            //         q_pos(k) += eps;
                    
            //         VectorX g_pos;
            //         evaluate_g_BDF1(q_pos, g_pos);

            //         H_fd.col(k) = (g_pos - g) / eps;
            //     }
            //     print_error_full("Simulation: H", H, H_fd);

            //     eps /= 10.;
            // }
            // set_state(q1, qdot1);
            // update_robot(save_backward_info && _backward_design_params_flag);

            
            // save backward data
            _backward_info._M.push_back(_M);
            _backward_info._D.push_back(_D);
            _backward_info._dfr_dqprev.push_back(_dfr_dqprev);
            _backward_info._dfr_dqdotprev.push_back(_dfr_dqdotprev);
            _backward_info._dg_du.push_back(-_h * _h * _dfr_du);
        }
    }
}

void Simulation::integration_BDF1(
        const VectorX q0, const VectorX qdot0, const dtype h, 
        VectorX& q1, VectorX& qdot1) {
    _q0 = q0;
    _qdot0 = qdot0;
    _h = h;

    q1 = q0 + h * qdot0; // initial guess
    newton(q1, &Simulation::evaluate_g_BDF1, &Simulation::evaluate_g_with_derivatives_BDF1);
    qdot1 = (q1 - q0) / h;

    // save backward data: H_inv, M, D, dg_du
    if (_backward_flag) {
        auto t_save_backward_start = clock();
        VectorX g;
        MatrixX H;
        evaluate_g_with_derivatives_BDF1(q1, g, H, true);

        _backward_info._H_lu.push_back(H.partialPivLu());

        _time_report._time_save_backward += clock() - t_save_backward_start;
    } 
    // else {
    //     VectorX g;
    //     evaluate_g_BDF1(q1, g);
    // }
}

void Simulation::evaluate_g_SDIRK2b(const VectorX& q1, VectorX& g) {
    dtype alpha = (2. - sqrt(2.)) / 2.;
    VectorX qdot1 = (q1 + (1. / alpha - 2.) * _q0 - (1. - alpha) / alpha * _q_alpha) / (alpha * _h);

    set_state(q1, qdot1);
    update_robot();

    computeMatrices(_M, _fr);

    g = _M * (q1 - _q0 - (2. * alpha - 1.) * _h * _qdot0 - 2. * (1. - alpha) * _h * _qdot_alpha) - alpha * alpha * _h * _h * _fr;
}

void Simulation::evaluate_g_with_derivatives_SDIRK2b(const VectorX& q1, VectorX& g, MatrixX& H, bool save_backward_info) {
    dtype alpha = (2. - sqrt(2.)) / 2.;
    VectorX qdot1 = (q1 + (1. / alpha - 2.) * _q0 - (1. - alpha) / alpha * _q_alpha) / (alpha * _h);
    
    set_state(q1, qdot1);
    update_robot(save_backward_info && _backward_design_params_flag);

    if (!save_backward_info) {
        computeMatrices(_M, _fr, _dM_dq, _K, _D);

        VectorX dq_tmp = q1 - _q0 - (2. * alpha - 1.) * _h * _qdot0 - 2. * (1. - alpha) * _h * _qdot_alpha;
        g = _M * dq_tmp - alpha * alpha * _h * _h * _fr;
        H = _M - alpha * alpha * _h * _h * _K - alpha * _h * _D;
        for (int k = 0;k < _ndof_r;k++) {
            H.col(k) += _dM_dq(k) * dq_tmp;
        }
    } else {
        if (_backward_design_params_flag) {
            computeMatrices(_M, _fr, _dM_dq, _K, _D, _dfr_du, _dM_dp, _dfr_dp);

            VectorX dq_tmp = q1 - _q0 - (2. * alpha - 1.) * _h * _qdot0 - 2. * (1. - alpha) * _h * _qdot_alpha;
            g = _M * dq_tmp - alpha * alpha * _h * _h * _fr;
            H = _M - alpha * alpha * _h * _h * _K - alpha * _h * _D;
            for (int k = 0;k < _ndof_r;k++) {
                H.col(k) += _dM_dq(k) * dq_tmp;
            }
            MatrixX dg_dp = -alpha * alpha * _h * _h * _dfr_dp;
            // for (int k = 0;k < _ndof_p;k++) {
            //     dg_dp.col(k) += dM_dp(k) * dq_tmp;
            // }
            for (int k = 0;k < _ndof_p1 + _ndof_p2;k++) {
                dg_dp.col(k) += _dM_dp(k) * dq_tmp;
            }
            for (int k = _ndof_p1 + _ndof_p2 + _ndof_p3;k < _ndof_p1 + _ndof_p2 + _ndof_p3 + _ndof_p4;k++) {
                dg_dp.col(k) += _dM_dp(k) * dq_tmp;
            }

            // save backward_data
            _backward_info._M.push_back(_M);
            _backward_info._D.push_back(_D);
            _backward_info._dg_dp.push_back(dg_dp);
            _backward_info._dg_du.push_back(-alpha * alpha * _h * _h * _dfr_du);
        } else {
            computeMatrices(_M, _fr, _dM_dq, _K, _D, _dfr_du);

            VectorX dq_tmp = q1 - _q0 - (2. * alpha - 1.) * _h * _qdot0 - 2. * (1. - alpha) * _h * _qdot_alpha;
            g = _M * dq_tmp - alpha * alpha * _h * _h * _fr;
            H = _M - alpha * alpha * _h * _h * _K - alpha * _h * _D;
            for (int k = 0;k < _ndof_r;k++) {
                H.col(k) += _dM_dq(k) * dq_tmp;
            }

            // save backward_data
            _backward_info._M.push_back(_M);
            _backward_info._D.push_back(_D);
            _backward_info._dg_du.push_back(-alpha * alpha * _h * _h * _dfr_du);
        }
    }
}

void Simulation::integration_SDIRK2(
        const VectorX q0, const VectorX qdot0, const dtype h, 
        VectorX& q1, VectorX& qdot1) {
    // step 1: 
    // solve for q_alpha
    // compute q_alpha and qdot_alpha by BDF1
    dtype alpha = (2. - sqrt(2.)) / 2.;
    VectorX q_alpha, qdot_alpha;
    
    integration_BDF1(q0, qdot0, alpha * h, q_alpha, qdot_alpha);

    // step 2:
    // compute q1 = q0 + (2alpha-1)qdot0 + 2(1-alpha)qdot_alpha + alpha^2*h^2*inv(M)*f(q1, qdot1)
    // q1_dot = (q1 + (1/a - 2) * q0 - (1 - a) / a * q_alpha) / (a * h)
    
    // save for newton's method
    _q0 = q0; _qdot0 = qdot0;
    _q_alpha = q_alpha; _qdot_alpha = qdot_alpha;
    _h = h;

    q1 = q_alpha + (1 - alpha) * h * qdot_alpha;

    newton(q1, &Simulation::evaluate_g_SDIRK2b, &Simulation::evaluate_g_with_derivatives_SDIRK2b);

    qdot1 = (q1 + (1. / alpha - 2.) * q0 - (1. - alpha) / alpha * q_alpha) / (alpha * h);

    // save backward data: H_inv, M, D, dg_du
    if (_backward_flag) {
        VectorX g;
        MatrixX H;
        auto t_save_backward_start = clock();
        evaluate_g_with_derivatives_SDIRK2b(q1, g, H, true);
        _backward_info._H_lu.push_back(H.partialPivLu());
        _time_report._time_save_backward += clock() - t_save_backward_start;
    } 
    // else {
    //     VectorX g;
    //     evaluate_g_SDIRK2b(q1, g);
    // }
}

void Simulation::evaluate_g_BDF2(const VectorX& q2, VectorX& g) {
    VectorX qdot2 = 3. / (2. * _h) * (q2 - 4. / 3. * _q1 + 1. / 3. * _q0);

    set_state(q2, qdot2);
    update_robot();

    computeMatrices(_M, _fr);

    g = _M * (q2 - 4. / 3. * _q1 + 1. / 3. * _q0 - 8. / 9. * _h * _qdot1 + 2. / 9. * _h * _qdot0) - 4. / 9. * _h * _h * _fr;
}

void Simulation::evaluate_g_with_derivatives_BDF2(const VectorX& q2, VectorX& g, MatrixX& H, bool save_backward_info) {
    VectorX qdot2 = 3. / (2. * _h) * (q2 - 4. / 3. * _q1 + 1. / 3. * _q0);

    set_state(q2, qdot2);
    update_robot(save_backward_info && _backward_design_params_flag);

    if (!save_backward_info) {
        computeMatrices(_M, _fr, _dM_dq, _K, _D);

        VectorX dq_tmp = q2 - 4. / 3. * _q1 + 1. / 3. * _q0 - 8. / 9. * _h * _qdot1 + 2. / 9. * _h * _qdot0;
        g = _M * dq_tmp - 4. / 9. * _h * _h * _fr;
        H = _M - 4. / 9. * _h * _h * _K - 2. / 3. * _h * _D;
        for (int k = 0;k < _ndof_r;k++) {
            H.col(k) += _dM_dq(k) * dq_tmp;
        }
    } else {
        if (_backward_design_params_flag) {
            computeMatrices(_M, _fr, _dM_dq, _K, _D, _dfr_du, _dM_dp, _dfr_dp);

            VectorX dq_tmp = q2 - 4. / 3. * _q1 + 1. / 3. * _q0 - 8. / 9. * _h * _qdot1 + 2. / 9. * _h * _qdot0;
            g = _M * dq_tmp - 4. / 9. * _h * _h * _fr;
            H = _M - 4. / 9. * _h * _h * _K - 2. / 3. * _h * _D;
            for (int k = 0;k < _ndof_r;k++) {
                H.col(k) += _dM_dq(k) * dq_tmp;
            }
            MatrixX dg_dp = -4./9. * _h * _h * _dfr_dp;
            for (int k = 0;k < _ndof_p1 + _ndof_p2;k++) {
                dg_dp.col(k) += _dM_dp(k) * dq_tmp;
            }
            for (int k = _ndof_p1 + _ndof_p2 + _ndof_p3;k < _ndof_p1 + _ndof_p2 + _ndof_p3 + _ndof_p4;k++) {
                dg_dp.col(k) += _dM_dp(k) * dq_tmp;
            }

            // save backward data
            _backward_info._M.push_back(_M);
            _backward_info._D.push_back(_D);
            _backward_info._dg_dp.push_back(dg_dp);
            _backward_info._dg_du.push_back(-4./9. * _h * _h * _dfr_du);
        } else {
            computeMatrices(_M, _fr, _dM_dq, _K, _D, _dfr_du);

            VectorX dq_tmp = q2 - 4. / 3. * _q1 + 1. / 3. * _q0 - 8. / 9. * _h * _qdot1 + 2. / 9. * _h * _qdot0;
            g = _M * dq_tmp - 4. / 9. * _h * _h * _fr;
            H = _M - 4. / 9. * _h * _h * _K - 2. / 3. * _h * _D;
            for (int k = 0;k < _ndof_r;k++) {
                H.col(k) += _dM_dq(k) * dq_tmp;
            }

            // save backward data
            _backward_info._M.push_back(_M);
            _backward_info._D.push_back(_D);
            _backward_info._dg_du.push_back(-4./9. * _h * _h * _dfr_du);
        }
    }
}

/*
u(k+1) = 4/3 * u(k) - 1/3 * u(k-1) + 2/3 * h * f(u(k+1))
*/
void Simulation::integration_BDF2(
        const VectorX q0, const VectorX qdot0, const VectorX q1, const VectorX qdot1, const dtype h, 
        VectorX& q2, VectorX& qdot2) {

    // save for newton's method
    _q0 = q0; _qdot0 = qdot0;
    _q1 = q1; _qdot1 = qdot1;
    _h = h;

    q2 = q1 + h * qdot1;

    newton(q2, &Simulation::evaluate_g_BDF2, &Simulation::evaluate_g_with_derivatives_BDF2);

    qdot2 = 3. / (2. * h) * (q2 - 4. / 3. * q1 + 1. / 3. * q0);
    
    // save backward data: H_inv, M, D, dg_du
    if (_backward_flag) {
        VectorX g;
        MatrixX H;
        auto t_save_backward_start = clock();
        evaluate_g_with_derivatives_BDF2(q2, g, H, true);
        _backward_info._H_lu.push_back(H.partialPivLu());
        _time_report._time_save_backward += clock() - t_save_backward_start;
    }
    // else {
    //     VectorX g;
    //     evaluate_g_BDF2(q2, g);
    // }
}

// backward related

// NOTE: set the variable indicators and set terminal derivatives in _backward_info
void Simulation::backward() {
    if (_q_his.size() <= 1) {
        std::cerr << "[Error] Please call simulation.forward() before simulation.backward()." << std::endl;
        throw "error";
    }
    
    int T = _q_his.size() - 1;

    // sanity checks
    if (_backward_info._flag_q0) {
        if (_backward_info._df_dq0.size() != _ndof_r) {
            throw_error("_backward_info._df_dq0.size != _ndof_r");
        }
    }
    if (_backward_info._flag_qdot0) {
        if (_backward_info._df_dqdot0.size() != _ndof_r) {
            throw_error("_backward_info._df_dqdot0.size != _ndof_r");
        }
    }
    if (_backward_info._flag_p) {
        if (_backward_info._df_dp.size() != _ndof_p) {
            throw_error("_backward_info._df_dp.size != _ndof_p");
        }
    }
    if (_backward_info._flag_u) {
        if (_backward_info._df_du.size() != _ndof_u * T) {
            throw_error("_backward_info._df_du.size != _ndof_u * T");
        }
    }
    if (_backward_info._df_dq.size() != _ndof_r * T) {
        throw_error("_backward_info._df_dq.size != _ndof_r * T");
    }
    if (_backward_info._df_dvar.size() != _ndof_var * T) {
        throw_error("_backward_info._df_dvar.size != _ndof_var * T");
    }
    if (_backward_info._df_dtactile.size() != _ndof_tactile * T) {
        throw_error("_backward_info._df_dtactile.size() != _ndof_tactile * T");
    }

    auto t_backward_start = clock();

    if (_options->_integrator == "BDF1") {
        backward_BDF1();
    } else if (_options->_integrator == "BDF2") {
        backward_BDF2();
    }

    _time_report._time_backward += clock() - t_backward_start;
}

void Simulation::backward_BDF1() {
    int T = _q_his.size() - 1;

    // initialize
    VectorX& df_dq = _backward_info._df_dq;
    VectorX& df_dvar = _backward_info._df_dvar;
    
    // backward propagation
    VectorX z = VectorX::Zero(T * _ndof_r);
    dtype h = _options->_h;
    
    for (int i = T;i >= 1;i--) {
        int k = i - 1;

        VectorX yk = df_dq.segment(k * _ndof_r, _ndof_r) + _backward_info._dvar_dq[k].transpose() * df_dvar.segment(k * _ndof_var, _ndof_var);

        if (_ndof_tactile > 0) {        
            // add the contribution from df_dtactile (dtactile_dq part)
            yk += _backward_info._dtactile_dq[k].transpose() * _backward_info._df_dtactile.segment(k * _ndof_tactile, _ndof_tactile);

            // add the contribution from df_dtactile (dtactile_dqdot part)
            // (1) from step k 
            yk += 1. / h * _backward_info._dtactile_dqdot[k].transpose() * _backward_info._df_dtactile.segment(k * _ndof_tactile, _ndof_tactile);
            // (2) from step k + 1
            if (k < T - 1) {
                yk -= 1. / h * _backward_info._dtactile_dqdot[k + 1].transpose() * _backward_info._df_dtactile.segment((k + 1) * _ndof_tactile, _ndof_tactile);
            }
        }

        if (k < T - 1) {
            // Add contributions from step k + 1
            MatrixX M = _backward_info._M[k + 1];
            MatrixX D = _backward_info._D[k + 1];
            MatrixX H = -2. * M + h * D - h * h * (_backward_info._dfr_dqprev[k + 1] + 1. / h * _backward_info._dfr_dqdotprev[k + 1]);
            yk -= H.transpose() * z.segment((k + 1) * _ndof_r, _ndof_r);
        }
        if (k < T - 2) {
            // Add contributions from step k + 2
            MatrixX H = _backward_info._M[k + 2] + h * _backward_info._dfr_dqdotprev[k + 2];
            yk -= H.transpose() * z.segment((k + 2) * _ndof_r, _ndof_r);
        }
        
        z.segment(k * _ndof_r, _ndof_r) = _backward_info._H_lu[k].transpose().solve(yk);
    }

    if (_backward_info._flag_q0) {
        // df_dq0 = _df_dq0 - dg_dq0' * z
        _backward_results._df_dq0 = _backward_info._df_dq0;

        // dg(1)_dq0 = -M(1) + hD(1)
        if (T > 0) {
            MatrixX dg1_dq0 = -_backward_info._M[0] + h * _backward_info._D[0] - h * h * _backward_info._dfr_dqprev[0];
            _backward_results._df_dq0 -= dg1_dq0.transpose() * z.head(_ndof_r);
        }

        // dg(2)_dq0 = M(2)
        if (T > 1) {
            MatrixX dg2_dq0 = _backward_info._M[1] + h * _backward_info._dfr_dqdotprev[1];
            _backward_results._df_dq0 -= dg2_dq0.transpose() * z.segment(_ndof_r, _ndof_r);
        }
        
        // add the contribution from df_dtactile (dtactile_dqdot(1) part)
        if (T > 0 && _ndof_tactile > 0) {
            _backward_results._df_dq0 -= 1. / h * _backward_info._dtactile_dqdot[0].transpose() * _backward_info._df_dtactile.head(_ndof_tactile);
        }
    }

    if (_backward_info._flag_qdot0) {
        // df_dqdot0 = _df_dqdot0 - dg_dqdot0' * z
        _backward_results._df_dqdot0 = _backward_info._df_dqdot0;
        
        // dg(1)_dqdot0 = -hM(1)
        if (T > 0) {
            MatrixX dg1_dqdot0 = -h * _backward_info._M[0] - h * h * _backward_info._dfr_dqdotprev[0];
            _backward_results._df_dqdot0 -= dg1_dqdot0.transpose() * z.head(_ndof_r);
        } 
    }
    
    if (_backward_info._flag_p) { 
        // df_dp = _df_dp - dg_dp' * z
        _backward_results._df_dp = _backward_info._df_dp;
        for (int k = 0;k < T;k++)
            _backward_results._df_dp += 
                - _backward_info._dg_dp[k].transpose() * z.segment(k * _ndof_r, _ndof_r)
                + _backward_info._dvar_dp[k].transpose() * _backward_info._df_dvar.segment(k * _ndof_var, _ndof_var);
    }

    if (_backward_info._flag_u) { 
        // df_du = _df_du - dg_du' * z
        _backward_results._df_du = _backward_info._df_du;
        for (int k = 0;k < T;k++)
            _backward_results._df_du.segment(k * _ndof_u, _ndof_u) -= 
                _backward_info._dg_du[k].transpose() * z.segment(k * _ndof_r, _ndof_r);
    }
}

// TODO: backward for tactile in BDF2

void Simulation::backward_BDF2() {
    int T = _q_his.size() - 1;

    dtype alpha = (2. - sqrt(2.)) / 2.;

    // initialize
    VectorX& df_dq = _backward_info._df_dq;
    VectorX& df_dvar = _backward_info._df_dvar;
    
    // backward propagation
    VectorX z = VectorX::Zero((T + 1) * _ndof_r);
    dtype h = _options->_h;
    for (int k = T;k >= 1;k--) {
        // VectorX yk = df_dq.segment((k - 1) * _ndof_r, _ndof_r) + _dvar_dq_his[k].transpose() * df_dvar.segment((k - 1) * _ndof_var, _ndof_var);
        VectorX yk = df_dq.segment((k - 1) * _ndof_r, _ndof_r) + _backward_info._dvar_dq[k - 1].transpose() * df_dvar.segment((k - 1) * _ndof_var, _ndof_var);
    
        if (k <= T - 1) {
            // Add contributions from step k + 1
            MatrixX M = _backward_info._M[k + 1];
            MatrixX D = _backward_info._D[k + 1];
            if (k > 1) { // BDF2 step    
                MatrixX H = -8. / 3. * M + 8. / 9. * h * D;
                yk -= H.transpose() * z.segment((k + 1) * _ndof_r, _ndof_r);
            } else {     // SDIRK2 step
                MatrixX H = -(8. / (9. * alpha) + 4. / 3.) * M + 8. / 9. * h * D;
                yk -= H.transpose() * z.segment((k + 1) * _ndof_r, _ndof_r);
            }
        }

        if (k <= T - 2) {
            // Add contributions from step k + 2
            MatrixX M = _backward_info._M[k + 2];
            MatrixX D = _backward_info._D[k + 2];
            if (k > 1) { // BDF2 step    
                MatrixX H = 22. / 9. * M - 2. / 9. * h * D;
                yk -= H.transpose() * z.segment((k + 2) * _ndof_r, _ndof_r);
            } else {     // SDIRK2 step
                MatrixX H = (2. / (9. * alpha) + 19. / 9.) * M - 2. / 9. * h * D;
                yk -= H.transpose() * z.segment((k + 2) * _ndof_r, _ndof_r);
            }
        }

        if (k <= T - 3) {
            // Add contributions from step k + 3
            MatrixX M = _backward_info._M[k + 3];
            MatrixX H = - 8. / 9. * M;
            yk -= H.transpose() * z.segment((k + 3) * _ndof_r, _ndof_r);
        }

        if (k <= T - 4) {
            // Add contributions from step k + 4
            MatrixX M = _backward_info._M[k + 4];
            MatrixX H = 1. / 9. * M;
            yk -= H.transpose() * z.segment((k + 4) * _ndof_r, _ndof_r);
        }

        z.segment(k * _ndof_r, _ndof_r) = _backward_info._H_lu[k].transpose().solve(yk);
    }

    // SDIRK2 (a) step
    VectorX ya = VectorX::Zero(_ndof_r); // qa is an internal state, so df_dqa = 0

    if (T >= 1) {
        MatrixX M = _backward_info._M[1];
        MatrixX D = _backward_info._D[1];
        MatrixX H = 2. * (alpha - 1.) / alpha * M + h * (1. - alpha) * D;
        ya -= H.transpose() * z.segment(_ndof_r, _ndof_r);
    }

    if (T >= 2) {
        MatrixX M = _backward_info._M[2];
        MatrixX H = 8. * (1. - alpha) / (9. * alpha * alpha) * M;
        ya -= H.transpose() * z.segment(_ndof_r * 2, _ndof_r);
    }

    if (T >= 3) {
        MatrixX M = _backward_info._M[3];
        MatrixX H = 2. * (alpha - 1.) / (9. * alpha * alpha) * M;
        ya -= H.transpose() * z.segment(_ndof_r * 3, _ndof_r);
    }

    // z.head(_ndof_r) = _backward_info._H_inv[0].transpose() * ya;
    z.head(_ndof_r) = _backward_info._H_lu[0].transpose().solve(ya);

    // aggregate everything together
    if (_backward_info._flag_q0) {
        // df_dq0 = _df_dq0 - dg_dq0' * z
        _backward_results._df_dq0 = _backward_info._df_dq0;

        if (T >= 1) {
            // dga_dq0 = -M + a * h * D
            MatrixX dga_dq0 = -_backward_info._M[0] + alpha * h * _backward_info._D[0];
            _backward_results._df_dq0 -= dga_dq0.transpose() * z.head(_ndof_r);
            // dg1_dq0 = (2-3a) / a * M - a * h * (1/a + 2) * D
            MatrixX dg1_dq0 = (2. - 3. * alpha) / alpha * _backward_info._M[1] - alpha * h * (1. / alpha - 2.) * _backward_info._D[1];
            _backward_results._df_dq0 -= dg1_dq0.transpose() * z.segment(_ndof_r, _ndof_r);
        }

        if (T >= 2) {
            // dg2_dq0 = (3a^2 + 16a - 8) / (9a^2) * M - 2/9 * h * D
            MatrixX dg2_dq0 = (3. * alpha * alpha + 16. * alpha - 8.) / (9. * alpha * alpha) * _backward_info._M[2]
                                - 2. / 9. * h * _backward_info._D[2];
            _backward_results._df_dq0 -= dg2_dq0.transpose() * z.segment(_ndof_r * 2, _ndof_r);
        }

        if (T >= 3) {
            // dg3_dq0 = -(4a^2 + 4a - 2) / (9a^2) * M
            MatrixX dg3_dq0 = - (4. * alpha * alpha + 4. * alpha - 2.) / (9. * alpha * alpha) * _backward_info._M[3];
            _backward_results._df_dq0 -= dg3_dq0.transpose() * z.segment(_ndof_r * 3, _ndof_r);
        }

        if (T >= 4) {
            // dg4_dq0 = 1/9 * M
            MatrixX dg4_dq0 = _backward_info._M[4] / 9.;
            _backward_results._df_dq0 -= dg4_dq0.transpose() * z.segment(_ndof_r * 4, _ndof_r);
        }
    }

    if (_backward_info._flag_qdot0) {
        // df_dqdot0 = _df_dqdot0 - dg_dqdot0' * z
        _backward_results._df_dqdot0 = _backward_info._df_dqdot0;
        
        if (T >= 1) {
            // dga_dqdot0 = -a * h * M
            MatrixX dga_dqdot0 = -alpha * h * _backward_info._M[0];
            _backward_results._df_dqdot0 -= dga_dqdot0.transpose() * z.head(_ndof_r);
            // dg1_dqdot0 = -(2a - 1) * h * M
            MatrixX dg1_dqdot0 = -(2. * alpha - 1.) * h * _backward_info._M[1];
            _backward_results._df_dqdot0 -= dg1_dqdot0.transpose() * z.segment(_ndof_r, _ndof_r);
        }

        if (T >= 2) {
            // dg2_dqdot0 = 2/9 * h * M
            MatrixX dg2_dqdot0 = 2. / 9. * h * _backward_info._M[2];
            _backward_results._df_dqdot0 -= dg2_dqdot0.transpose() * z.segment(_ndof_r * 2, _ndof_r);
        }
    }

    if (_backward_info._flag_p) {
        // df_dp = _df_dp - dg_dp' * z
        _backward_results._df_dp = _backward_info._df_dp;
        for (int k = 0;k <= T;k++)
            _backward_results._df_dp -= 
                _backward_info._dg_dp[k].transpose() * z.segment(k * _ndof_r, _ndof_r);
        for (int k = 0;k < T;k++)
            _backward_results._df_dp += 
                _backward_info._dvar_dp[k].transpose() * _backward_info._df_dvar.segment(k * _ndof_var, _ndof_var);
    }

    if (_backward_info._flag_u) {
        // df_du = _df_du - dg_du' * z
        _backward_results._df_du = _backward_info._df_du;
        for (int k = 0;k < T;k++)
            _backward_results._df_du.segment(k * _ndof_u, _ndof_u) -= 
                _backward_info._dg_du[k + 1].transpose() * z.segment((k + 1) * _ndof_r, _ndof_r);
        _backward_results._df_du.head(_ndof_u) -= _backward_info._dg_du[0].transpose() * z.head(_ndof_r);
    }
}

void Simulation::backward_steps(int num_backward_steps) {
    if (_backward_info._current_backward_step <= 0) {
        std::cerr << "[Error] Please call simulation.forward() before simulation.backward()." << std::endl;
        throw "error";
    }

    // sanity checks
    // TODO: support for q0 and qdot0
    if (_backward_info._flag_q0) {
        throw_error("_backward_info._flag_q0 should be false for backward_steps");
    }
    if (_backward_info._flag_qdot0) {
        throw_error("_backward_info._flag_qdot0 should be false for backward_steps");
    }
    if (_backward_info._flag_p) {
        if (_backward_info._df_dp.size() != _ndof_p) {
            throw_error("_backward_info._df_dp.size != _ndof_p");
        }
    }
    if (_backward_info._flag_u) {
        if (_backward_info._df_du.size() != _ndof_u * num_backward_steps) {
            throw_error("_backward_info._df_du.size != _ndof_u * num_backward_steps");
        }
    }
    if (_backward_info._df_dq.size() != _ndof_r * num_backward_steps) {
        throw_error("_backward_info._df_dq.size != _ndof_r * num_backward_steps");
    }
    if (_backward_info._df_dvar.size() != _ndof_var * num_backward_steps) {
        throw_error("_backward_info._df_dvar.size != _ndof_var * num_backward_steps");
    }
    if (_backward_info._df_dtactile.size() != _ndof_tactile * num_backward_steps) {
        throw_error("_backward_info._df_dtactile.size() != _ndof_tactile * num_backward_steps");
    }

    auto t_backward_start = clock();

    if (_options->_integrator == "BDF1") {
        backward_steps_BDF1(num_backward_steps);
    } else if (_options->_integrator == "BDF2") {
        throw_error("backward_steps has not supported BDF2 yet.");
    }

    _time_report._time_backward += clock() - t_backward_start;
}

void Simulation::backward_steps_BDF1(int num_backward_steps) {
    // initialize
    VectorX& df_dq = _backward_info._df_dq;
    VectorX& df_dvar = _backward_info._df_dvar;
    
    dtype h = _options->_h;
    int T = _backward_info._q_his.size() - 1;

    if (_backward_info._flag_u) {
        _backward_results._df_du = _backward_info._df_du;
    }

    for (int i = num_backward_steps;i >= 1;--i) {
        int k_real = _backward_info._current_backward_step - num_backward_steps + i - 1;
        int k = i - 1;
        VectorX yk = df_dq.segment(k * _ndof_r, _ndof_r) + _backward_info._dvar_dq[k_real].transpose() * df_dvar.segment(k * _ndof_var, _ndof_var);
        // add the contribution from df_dtactile (dtactile_dq part)
        yk += _backward_info._dtactile_dq[k_real].transpose() * _backward_info._df_dtactile.segment(k * _ndof_tactile, _ndof_tactile);

        // add the contribution from df_dtactile (dtactile_dqdot part)
        // (1) from step k 
        yk += 1. / h * _backward_info._dtactile_dqdot[k_real].transpose() * _backward_info._df_dtactile.segment(k * _ndof_tactile, _ndof_tactile);
        // (2) from step k + 1
        if (k_real < T - 1) {
            yk -= 1. / h * _backward_info._dtactile_dqdot[k_real + 1].transpose() * _backward_info._df_dtactile_his[_backward_info._df_dtactile_his.size() - 1];
        }

        // add the contribution from computed df_dq 
        if (k_real < T - 1) {
            // Add contributions from step k + 1
            MatrixX M = _backward_info._M[k_real + 1];
            MatrixX D = _backward_info._D[k_real + 1];
            MatrixX H = -2. * M + h * D - h * h * (_backward_info._dfr_dqprev[k_real + 1] + 1. / h * _backward_info._dfr_dqdotprev[k_real + 1]);
            yk -= H.transpose() * _backward_info._z[_backward_info._z.size() - 1];
        }
        if (k_real < T - 2) {
            // Add contributions from step k + 2
            MatrixX H = _backward_info._M[k_real + 2] + h * _backward_info._dfr_dqdotprev[k_real + 2];
            yk -= H.transpose() * _backward_info._z[_backward_info._z.size() - 2];
        }
        _backward_info._z.push_back(_backward_info._H_lu[k_real].transpose().solve(yk));

        if (_backward_info._flag_u) {
            _backward_results._df_du.segment(k * _ndof_u, _ndof_u) -=
                _backward_info._dg_du[k_real].transpose() * _backward_info._z[_backward_info._z.size() - 1];
        }
        _backward_info._df_dtactile_his.push_back(_backward_info._df_dtactile.segment(k * _ndof_tactile, _ndof_tactile));
    }

    _backward_info._current_backward_step -= num_backward_steps;
}

// viewer related

void Simulation::get_rendering_objects(
    std::vector<Matrix3Xf>& vertex_list, 
    std::vector<Matrix3Xi>& face_list,
    std::vector<opengl_viewer::Option>& option_list,
    std::vector<opengl_viewer::Animator*>& animator_list) {
    
    _robot->get_rendering_objects(vertex_list, face_list, option_list, animator_list);
}

void Simulation::init_viewer() {
    if (!_viewer) {
        _viewer = std::make_shared<SimViewer>(this);
    }
    _viewer->initialize();
    _viewer_step = 0;
    set_state(_q_his[_viewer_step], _qdot_his[_viewer_step]);
    update_robot();
    _robot->set_virtual_object_data(_virtual_object_data_his[_viewer_step]);
}

// advance the viewer steps, and return whether it completes one trajectory.
bool Simulation::advance_viewer_step(int num_steps) {
    _viewer_step += num_steps;

    bool done = false;

    if (_viewer_step >= _q_his.size()) {
        done = true;
        if (_viewer_options->_loop) {
            _viewer_step %= _q_his.size();
        } else {
            _viewer_step = _q_his.size() - 1;
        }
    }

    set_state(_q_his[_viewer_step], _qdot_his[_viewer_step]);
    update_robot();
        
    _robot->set_virtual_object_data(_virtual_object_data_his[_viewer_step]);

    return done;
}

void Simulation::replay() {
    if (_q_his.size() == 0) {
        std::cerr << "[Error] Please call simulation.reset() before simulation.run_viewer()." << std::endl;
        throw "error";
    }
    // save the current q and qdot
    VectorX q = get_q();
    VectorX qdot = get_qdot();
    VectorX virtual_object_data = _robot->get_virtual_object_data();
    init_viewer();
    _viewer->run();

    // restore the q and qdot
    set_state(q, qdot);
    update_robot();
    
    _robot->set_virtual_object_data(virtual_object_data);
}

void Simulation::export_replay(std::string folder) {
    replay();

    // export the meshes
    int idx = 0;
    
    for (auto body : _robot->_bodies) {
        std::string path = folder + "/meshes/" + to_string(idx) + ".obj";
        opengl_viewer::WriteToObjFile(body->_rendering_vertices, body->_rendering_faces, path);
        idx ++;
    }

    for (auto virtual_object : _robot->_virtual_objects) {
        std::string path = folder + "/meshes/" + to_string(idx) + ".obj";
        opengl_viewer::WriteToObjFile(virtual_object->_rendering_vertices, virtual_object->_rendering_faces, path);
        idx ++;
    }

    for (auto tactile_sensor : _robot->_tactile_sensors) {
        std::string path = folder + "/meshes/" + to_string(idx) + ".obj";
        std::cerr << tactile_sensor->_rendering_vertices.cols() << " " << tactile_sensor->_rendering_faces.cols() << std::endl;
        opengl_viewer::WriteToObjFile(tactile_sensor->_rendering_vertices, tactile_sensor->_rendering_faces, path);
        idx ++;
    }

    for (auto end_effector : _robot->_end_effectors) {
        if (end_effector->_radius > 0.) {
            std::string path = folder + "/meshes/" + to_string(idx) + ".obj";
            opengl_viewer::WriteToObjFile(end_effector->_rendering_vertices, end_effector->_rendering_faces, path);
            idx ++;
        }
    }

    // export the transformation matrices
    for (int i = 0;i < _q_his.size();i++) {
        set_state(_q_his[i], _qdot_his[i]);
        update_robot();
        std::string path = folder + "/" + to_string(i) + ".txt";
        FILE* fp = fopen(path.c_str(), "w");
        fprintf(fp, "%d\n", idx);
        for (auto body : _robot->_bodies) {
            for (int j = 0;j < 4;j++) {
                for (int k = 0;k < 4;k++) {
                    fprintf(fp, "%.6lf ", body->_E_0i(j, k));
                }
                fprintf(fp, "\n");
            }
        }
        for (auto virtual_object : _robot->_virtual_objects) {
            Matrix4 E = virtual_object->get_transform_matrix();
            for (int j = 0;j < 4;j++) {
                for (int k = 0;k < 4;k++) {
                    fprintf(fp, "%.6lf ", E(j, k));
                }
                fprintf(fp, "\n");
            }
        }
        for (auto tactile_sensor : _robot->_tactile_sensors) {
            Matrix4 E = tactile_sensor->_body->_E_0i;
            for (int j = 0;j < 4;j++) {
                for (int k = 0;k < 4;k++) {
                    fprintf(fp, "%.6lf ", E(j, k));
                }
                fprintf(fp, "\n");
            }
        }
        for (auto end_effector : _robot->_end_effectors) {
            if (end_effector->_radius > 0.) {
                Matrix4 E = end_effector->_joint->_E_0j;
                Vector3 pos_world = end_effector->_joint->position_in_world(end_effector->_pos);
                E.topRightCorner(3, 1) = pos_world;

                for (int j = 0;j < 4;j++) {
                    for (int k = 0;k < 4;k++) {
                        fprintf(fp, "%.6lf ", E(j, k));
                    }
                    fprintf(fp, "\n");
                }
            }
        }

        fclose(fp);
    }
}

void Simulation::print_time_report() {
    std::cerr << "----------- Time Report -----------" << std::endl;
    std::cerr << "|Simulation                       |" << std::endl;
    std::cerr << "|---------------------------------|" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "Solver" << "|" << std::setw(7) << std::right << _time_report._time_solver / 1000 << "(ms) |" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "Compute Matrices" << "|" << std::setw(7) << std::right << _time_report._time_compute_matrices / 1000 << "(ms) |" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "-- Compute force" << "|" << std::setw(7) << std::right << _time_report._time_compute_force / 1000 << "(ms) |" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "-- Compute dJ" << "|" << std::setw(7) << std::right << _time_report._time_compute_dJ / 1000 << "(ms) |" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "-- Compute df" << "|" << std::setw(7) << std::right << _time_report._time_compute_df / 1000 << "(ms) |" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "Compose Matrices" << "|" << std::setw(7) << std::right << _time_report._time_compose_matrices / 1000 << "(ms) |" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "-- Compose dM_dp" << "|" << std::setw(7) << std::right << _time_report._time_dM_dp / 1000 << "(ms) |" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "---- Compose dM_dp1" << "|" << std::setw(7) << std::right << _time_report._time_dM_dp1 / 1000 << "(ms) |" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "---- Compose dM_dp2" << "|" << std::setw(7) << std::right << _time_report._time_dM_dp2 / 1000 << "(ms) |" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "---- Compose dM_dp4" << "|" << std::setw(7) << std::right << _time_report._time_dM_dp4 / 1000 << "(ms) |" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "-- Compose df_dp" << "|" << std::setw(7) << std::right << _time_report._time_df_dp / 1000 << "(ms) |" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "Save Backward" << "|" << std::setw(7) << std::right << _time_report._time_save_backward / 1000 << "(ms) |" << std::endl;
    std::cerr << "|" << std::setw(20) << std::left << "Backward" << "|" << std::setw(7) << std::right << _time_report._time_backward / 1000 << "(ms) |" << std::endl;
    // _robot->print_time_report();
    std::cerr << "-----------------------------------" << std::endl;
}

}