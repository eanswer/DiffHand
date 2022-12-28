#pragma once
#include "Common.h"
#include "Utils.h"
#include "BackwardData.h"
#include "pugixml.hpp"

namespace redmax {

class Robot;
class SimViewer;
class Joint;
class Body;
class Tactile;

class Simulation {
private:
    // state history
    std::vector<VectorX> _q_his, _qdot_his;
    std::vector<VectorX> _virtual_object_data_his;

    // utils
    std::vector<Vector3> parse_contact_points(std::string str);

    /** compute simulation-related matrices/vectors and derivatives **/
    // compaute M and f
    void computeMatrices(MatrixX& M, VectorX& fr); // evaluate force and mass matrix
    void computeMatrices(MatrixX& M, VectorX& fr, JacobianMatrixVector& dM_dq, MatrixX& K, MatrixX& D); // evaluate derivatives to q
    void computeMatrices(MatrixX& M, VectorX& fr, JacobianMatrixVector& dM_dq, MatrixX& K, MatrixX& D, MatrixX& dfr_du); // evaluate derivatives to q and to optimization parameters
    void computeMatrices(
        MatrixX& M, VectorX& fr,
        JacobianMatrixVector& dM_dq, MatrixX& K, MatrixX& D,
        MatrixX& dfr_du,
        JacobianMatrixVector& dM_dp, MatrixX& df_dp); // evaluate derivatives to q and to optimization parameters
    
    // derivatives w.r.t. previous states, e.g. from position controlled actuator, now only for fr
    void computeExtraDerivative(MatrixX& dfr_dqprev, MatrixX& dfr_dqdotprev);
    
    // compute M(q)x + y*f(q) using jacobian product
    void computeMatricesProduct(const VectorX& x, const dtype y, VectorX& g); // g = M(q) * x + y * f(q)
    void computeMatricesProduct(const VectorX& x, const dtype y, VectorX& g, MatrixX& H_sub, MatrixX& M, MatrixX& K, MatrixX& D); // g = M(q) * x + y * f(q), H_sub.col(k) = dM_dq(k) * x

    // compute variables
    void computeVariablesWithDerivative(VectorX& variables, MatrixX& dvar_dq);
    void computeVariablesWithDerivative(VectorX& variables, MatrixX& dvar_dq, MatrixX& dvar_dp);

    // compute tactile
    void computeTactileWithDerivatives(VectorX& tactile_force, MatrixX& dtactile_dq, MatrixX& dtactile_dqdot);

    // temporary variables for evaluation functions
    VectorX _q1, _qdot1, _q0, _qdot0, _q_alpha, _qdot_alpha;
    dtype _h;

    // reserved variables to be filled
    MatrixX _J, _Jdot;
    JacobianMatrixVector _dJ_dq, _dJdot_dq;
    VectorX _Mm;
    VectorX _fm, _fr;
    MatrixX _Km, _Dm, _Kr, _Dr;
    MatrixX _dfm_du;
    MatrixX _dphi_dq; // dphi_dq = outer_product(dJ_dq, qdot)
    
    MatrixX _M, _K, _D, _dfr_du;
    MatrixX _dfr_dqprev, _dfr_dqdotprev;
    // VectorX _f;
    JacobianMatrixVector _dM_dq;

    JacobianMatrixVector _dM_dp;
    MatrixX _dfr_dp;

    // Viewer
    std::shared_ptr<SimViewer> _viewer;
    int _viewer_step;

    void evaluate_g_BDF1(const VectorX& q1, VectorX& g);
    void evaluate_g_with_derivatives_BDF1(const VectorX& q1, VectorX& g, MatrixX& H, bool save_backward_info = false);
    void evaluate_g_SDIRK2b(const VectorX& q1, VectorX& g);
    void evaluate_g_with_derivatives_SDIRK2b(const VectorX& q1, VectorX& g, MatrixX& H, bool save_backward_info = false);
    void evaluate_g_BDF2(const VectorX& q2, VectorX& g);
    void evaluate_g_with_derivatives_BDF2(const VectorX& q2, VectorX& g, MatrixX& H, bool save_backward_info = false);

    // newton's method
    typedef void (Simulation::*Func)(const VectorX&, VectorX&);
    typedef void (Simulation::*Func_With_Derivatives)(const VectorX&, VectorX&, MatrixX&, bool);
    void newton(VectorX& q, Func func, Func_With_Derivatives func_with_derivatives);

    // integration methods
    void integration_BDF1(const VectorX q0, const VectorX qdot0, const dtype h, VectorX& q1, VectorX& qdot1);
    void integration_SDIRK2(const VectorX q0, const VectorX qdot0, const dtype h, VectorX& q1, VectorX& qdot1);
    void integration_BDF2(const VectorX q0, const VectorX qdot0, const VectorX q1, const VectorX qdot1, const dtype h, VectorX& q2, VectorX& qdot2);

    // backward methods
    void backward_BDF1();
    void backward_BDF2();
    
    // backward steps methods
    void backward_steps_BDF1(int num_backward_steps);
    
    // xml file related
    std::string _asset_folder;
    
public:
    // -------------------- Constants -----------------------
    class Options {
    public:
        Vector3 _gravity;
        dtype _h;
        string _integrator; // [euler, rk4, BDF1, BDF2]
        string _unit; // ["cm-g", "m-kg"]

        Options(Vector3 gravity = -980. * Vector3::UnitZ(), dtype h = 0.02, string integrator = "BDF1", string unit = "cm-g"): 
            _gravity(gravity), _h(h), _integrator(integrator), _unit(unit) {}
    };
    
    Options *_options;

    class ViewerOptions {
    public:
        int _fps;
        dtype _speed;
        Vector3 _camera_pos;
        Vector3 _camera_up;
        Vector3 _camera_lookat;
        Vector4 _background_color_rgba;
        bool _ground;
        Matrix4 _E_g;
        bool _record;
        std::string _record_folder;
        bool _loop;     // whether loop the replay
        bool _infinite; // whether to replay until close the window

        ViewerOptions() {
            _fps = 30;
            _speed = 1.;
            _background_color_rgba = Vector4(0., 0., 0., 1.);
            _camera_pos = Vector3(4., -5., 3.);
            _camera_up = Vector3::UnitZ();
            _camera_lookat = Vector3::Zero();
            _ground = true;
            _E_g.topLeftCorner(3, 3).setIdentity();
            _E_g.topRightCorner(3, 1) = Vector3::UnitZ() * -2.;
            _record = false;
            _loop = true;
            _infinite = true;
        }
    };

    ViewerOptions * _viewer_options;

    class SolverOptions {
    public:
        dtype _tol;
        int _MaxIter_Newton;
        int _MaxIter_LS;
        int _MaxIter_LS_Fail_Strike; // not used

        SolverOptions() {
            _tol = 1e-9;
            _MaxIter_Newton = 100;
            _MaxIter_LS = 20;
            _MaxIter_LS_Fail_Strike = 5;
        }
    };

    SolverOptions * _solver_options;

    class TimeReport {
    public:
        long long _time_solver, _time_save_backward, _time_backward;
        long long _time_compute_matrices, _time_compose_matrices;
        long long _time_compute_jacobian, _time_compute_force, _time_compute_jacobian_derivatives;
        long long _time_compute_dJ, _time_compute_df;
        long long _time_dM_dp, _time_df_dp;
        long long _time_dM_dp1, _time_dM_dp2, _time_dM_dp4;

        void reset() {
            _time_solver = _time_save_backward = _time_backward = 0;
            _time_compute_matrices = _time_compose_matrices = 0;
            _time_compute_dJ = _time_compute_df = 0;
            _time_dM_dp = _time_df_dp = 0;
            _time_dM_dp1 = _time_dM_dp2 = _time_dM_dp4 = 0;
            _time_compute_jacobian = _time_compute_force = _time_compute_jacobian_derivatives = 0;
        }
    };

    TimeReport _time_report;

    // -------------------- forward dynamics related -------------------
    std::string _name;

    // robot related
    Robot* _robot;
    std::map<std::string, Joint*> _joint_map;
    std::map<std::string, Body*> _body_map;

    bool _ground;
    Matrix4 _E_g; // ground transform matrix

    int _ndof_r, _ndof_m, _ndof_u, _ndof_var, _ndof_tactile;
    int _ndof_p, _ndof_p1, _ndof_p2, _ndof_p3, _ndof_p4, _ndof_p5, _ndof_p6;

    // controller parameters
    VectorX _phi;

    // states
    VectorX _q_init, _qdot_init;

    // backward related
    bool _backward_flag, _backward_design_params_flag; // whether do backward after forward
    BackwardInfo _backward_info;
    BackwardResults _backward_results;
    std::vector<BackwardInfo> _backward_info_his;
    std::vector<BackwardResults> _backward_results_his;

    // verbose output
    bool _verbose;

    // constructors
    Simulation(Options *options, std::string name = "");
    Simulation(Options *options, ViewerOptions *viewer_options, std::string name = "");
    Simulation(std::string xml_file_path, bool verbose = false);
    
    // destructor
    ~Simulation();

    Joint* parse_from_xml_file(pugi::xml_node root, pugi::xml_node node, \
                                Joint* parent_joint, int &joint_cnt, bool verbose = false);

    void addRobot(Robot* robot) {
        _robot = robot;
    }

    // init simulation
    void init(bool verbose = false);    

    // init states
    void set_state_init(const VectorX q_init, const VectorX qdot_init);
    void set_q_init(const VectorX q_init);
    void set_qdot_init(const VectorX qdot_init);
    const VectorX get_q_init();
    const VectorX get_qdot_init();

    // states
    void set_state(const VectorX q, const VectorX qdot);
    void set_q(const VectorX q);
    void set_qdot(const VectorX qdot);
    const VectorX get_q();
    const VectorX get_qdot();
    void reparam();

    // variables
    const VectorX get_variables();

    // control variables
    void set_u(const VectorX& u);
    void get_ctrl_range(VectorX& ctrl_min, VectorX& ctrl_max);
    VectorX get_ctrl_force();
    void print_ctrl_info();

    // tactile related functions
    std::vector<Vector3> get_tactile_sensor_pos(std::string name);
    std::vector<Vector2i> get_tactile_image_pos(const string name);
    std::vector<dtype> get_tactile_depth(const string name);
    std::vector<dtype> get_tactile_normal_force(const string name);
    std::vector<Vector2> get_tactile_shear_force(const string name);
    std::vector<Vector3> get_tactile_force(const string name);
    VectorX get_tactile_force_vector();
    std::vector<std::vector<std::vector<Vector3>>> get_tactile_flow_images();

    // design parameters
    void set_design_params(const VectorX &design_params);
    VectorX get_design_params();
    void print_design_params_info();

    // set contact coefficient scale
    void set_contact_scale(dtype scale);

    // rendering mesh for abstract bodies
    void set_rendering_mesh_vertices(const std::vector<Matrix3X> &Vs);
    void set_rendering_mesh(const std::vector<Matrix3X> &Vs, const std::vector<Matrix3Xi> &Fs);

    // virtual objects
    void update_virtual_object(std::string name, VectorX data);

    // updata robot to propagate the state
    void update_robot(bool design_gradient = false);

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

    void test_derivatives_runtime();
    void test_design_derivatives_runtime();

    // reset the simulation
    void reset(bool backward_flag = false, bool backward_design_params_flag = false);
    
    // manipulate history cache for backward info and backward results
    void clearBackwardCache();
    void saveBackwardCache();
    void popBackwardCache();
    int backwardCacheSize();

    // verbose defines if log state history
    void forward(int num_steps, bool verbose = false, bool test_derivatives = false, bool save_last_frame_var_only = false);
    
    // backward to compute the derivatives
    void backward();
        
    // backward several steps
    void backward_steps(int num_backward_steps);

    // rendering
    void get_rendering_objects(
            std::vector<Matrix3Xf>& vertex_list, 
            std::vector<Matrix3Xi>& face_list,
            std::vector<opengl_viewer::Option>& option_list,
            std::vector<opengl_viewer::Animator*>& animator_list);

    void init_viewer();
    bool advance_viewer_step(int num_steps);
    void replay();

    // export simulation replay to a folder
    void export_replay(std::string folder);

    void print_time_report();
};

}