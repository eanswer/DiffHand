#pragma once
#include "Common.h"
#include "Utils.h"

namespace redmax {

class Joint;
class Body;
class TactileSensor;
class Spring;
class Force;
class Actuator;
class EndEffector;
class VirtualObject;

class Robot {
public:
    vector<Joint*> _joints;
    vector<Body*> _bodies;
    vector<Force*> _forces;
    vector<Actuator*> _actuators;
    vector<EndEffector*> _end_effectors;
    vector<TactileSensor*> _tactile_sensors;
    vector<VirtualObject*> _virtual_objects;
    vector<Joint*> _root_joints;
    int _ndof_r, _ndof_m, _ndof_u, _ndof_var, _ndof_virtual, _ndof_tactile;
    int _ndof_p, _ndof_p1, _ndof_p2, _ndof_p3, _ndof_p4, _ndof_p5, _ndof_p6; // design parameters
    // reserve for dfm_dp and dfr_dp
    MatrixX dfm_dp, dfr_dp;
    // reserve for dJ_dp1, dJ_dp2, dJdot_dp1, dJdot_dp2
    JacobianMatrixVector dJ_dp1, dJ_dp2, dJdot_dp1, dJdot_dp2;
    // tactile force container
    VectorX _tactile_force;

    Robot() {
        _joints.clear();
        _bodies.clear();
        _forces.clear();
        _root_joints.clear();
        _actuators.clear();
        _end_effectors.clear();
    }

    void add_tactile_sensor(TactileSensor* tactile_sensor);

    void add_force(Force* force);
    
    void add_actuator(Actuator* actuator);

    void add_end_effector(EndEffector* end_effector);

    void add_virtual_object(VirtualObject* virtual_object);

    // init robot
    void init(bool verbose = false);
    void construct_dfs_order(Joint* now);

    // update robot
    void update(bool design_gradient = false);

    // check if state is valid, invalid means reparameterization needed
    bool check_valid();
    
    // set state variables
    void set_q(const VectorX q);
    void set_qdot(const VectorX qdot);
    void set_state(const VectorX q, const VectorX qdot);
    
    // get state variables
    VectorX get_q();
    VectorX get_qdot();
    VectorX get_phi();
    VectorX get_variables();

    // reparameterization
    void reparam();

    // control variables
    void update_actuator_states(const VectorX& q_new, const VectorX& qdot_new);
    VectorX get_u();
    void set_u(const VectorX& u);
    void get_ctrl_range(VectorX& ctrl_min, VectorX& ctrl_max);
    VectorX get_ctrl_force();
    void print_ctrl_info();

    // tactile related user functions
    std::vector<Vector3> get_tactile_sensor_pos(std::string name);
    std::vector<Vector2i> get_tactile_image_pos(std::string name);
    std::vector<dtype> get_tactile_depth(std::string name);
    std::vector<dtype> get_tactile_normal_force(const string name);
    std::vector<Vector2> get_tactile_shear_force(const string name);
    std::vector<Vector3> get_tactile_force(const string name);
    VectorX get_tactile_force_vector();
    std::vector<std::vector<std::vector<Vector3>>> get_tactile_flow_images();
    
    // design parameters
    VectorX get_design_params();
    void set_design_params(const VectorX& design_params);
    void print_design_params_info();

    // set contact coefficient scale
    void set_contact_scale(dtype scale);

    // rendering mesh for abstract bodies
    void set_rendering_mesh_vertices(const std::vector<Matrix3X> &Vs);
    void set_rendering_mesh(const std::vector<Matrix3X> &Vs, const std::vector<Matrix3Xi> &Fs);

    // virtual object
    void update_virtual_object(std::string name, VectorX data);
    VectorX get_virtual_object_data();
    void set_virtual_object_data(const VectorX& q);

    // parameter update
    void update_contact_parameters(std::string body1, std::string body2, dtype kn, dtype kt, dtype mu, dtype damping);
    void update_tactile_parameters(std::string name, dtype kn, dtype kt, dtype mu, dtype damping);
    void update_body_density(std::string body_name, dtype density);
    void update_body_color(std::string body_name, Vector3 color);
    void update_body_size(std::string body_name, VectorX body_size);
    void update_joint_damping(std::string joint_name, dtype damping);
    void update_tactile_sensor_pos(std::string name, std::vector<Vector3> &new_pos);
    void update_joint_location(std::string joint_name, Vector3 joint_location);
    void update_endeffector_position(std::string endeffector_name, Vector3 position);
    
    // compute simulation-related variables and derivatives
    void computeMaximalMassMatrix(VectorX& Mm);
    void computeForce(VectorX& fm, VectorX& fr);
    void computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr, bool verbose = false);
    void computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr, MatrixX& dfm_dp, MatrixX& dfr_dp, bool verbose = false);
    void computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr, SparseMatrixX& dfm_dp_sparse, SparseMatrixX& dfr_dp_sparse, bool verbose = false);
    void computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr, MatrixX& dfm_dp, SparseMatrixX& dfr_dp_sparse, bool verbose = false);
    void computeJointJacobian(MatrixX& J, MatrixX& Jdot);
    void computeJointJacobianWithDerivative(MatrixX& J, MatrixX& Jdot, JacobianMatrixVector& dJ_dq, JacobianMatrixVector& dJdot_dq);
    void computeJointJacobianWithDerivative(
        MatrixX& J, MatrixX& Jdot, 
        JacobianMatrixVector& dJ_dq, JacobianMatrixVector& dJdot_dq,
        JacobianMatrixVector& dJ_dp1, JacobianMatrixVector& dJ_dp2,
        JacobianMatrixVector& dJdot_dp1, JacobianMatrixVector& dJdot_dp2);
    void computeJointJacobianWithDerivative(
        MatrixX& J, MatrixX& Jdot, 
        JacobianMatrixVector& dJ_dq, JacobianMatrixVector& dJdot_dq,
        SparseJacobianMatrixVector& dJ_dp1_sparse, SparseJacobianMatrixVector& dJ_dp2_sparse,
        SparseJacobianMatrixVector& dJdot_dp1_sparse, SparseJacobianMatrixVector& dJdot_dp2_sparse);
    void computeJointJacobianProduct(const VectorX& x, VectorX& Jx);
    void computeJointJacobianDotProduct(const VectorX& x, VectorX& Jdotx);
    void computeJointJacobianTransposeProduct(const VectorX& y, VectorX& JTy);
    void computeJointJacobianDerivativeProduct(const VectorX& x, MatrixX& dJdq_x);
    void computeJointJacobianDotDerivativeProduct(const VectorX& x, MatrixX& dJdotdq_x);
    void computeJointJacobianTransposeDerivativeProduct(const VectorX& x, MatrixX& dJTdq_x);
        
    // compute df_du
    void compute_dfdu(MatrixX& dfm_du, MatrixX& dfr_du);

    // compute affiliated variables and derivatives
    void computeVariables(VectorX& variables);
    void computeVariablesWithDerivative(VectorX& variables, MatrixX& dvar_dq);
    void computeVariablesWithDerivative(VectorX& variables, MatrixX& dvar_dq, MatrixX& dvar_dp);
    
    // tactile forces and derivatives
    void computeTactileWithDerivatives(VectorX& tactile_force, MatrixX& dtactile_dqm, MatrixX& dtactile_dphi, std::vector<std::pair<Body*, Body*>>& contact_bodies);

    // derivatives w.r.t. previous states, e.g. from position controlled actuator, now only for fr
    void computeExtraDerivative(MatrixX& dfr_dqprev, MatrixX& dfr_dqdotprev);
    
    // test derivatives
    void test_derivatives_runtime();
    void test_design_derivatives_runtime();

    
    // rendering related
    void get_rendering_objects(
        std::vector<Matrix3Xf>& vertex_list, 
        std::vector<Matrix3Xi>& face_list,
        std::vector<opengl_viewer::Option>& option_list,
        std::vector<opengl_viewer::Animator*>& animator_list);
    
    void reset_time_report();
    void print_time_report();
};

}