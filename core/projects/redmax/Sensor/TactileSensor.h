// TODO: clean up
#pragma once
#include "Common.h"
#include "Utils.h"

namespace redmax {

class Body;
class Robot;
class TactileSensor;

class TactileSensorAnimator : public opengl_viewer::Animator {
public:
    TactileSensorAnimator(TactileSensor* tactile_sensor) : _tactile_sensor(tactile_sensor) {}

    const Eigen::Matrix4f AnimatedModelMatrix(const float t);

    TactileSensor* _tactile_sensor;
};

class TactileSensor {
public:
    TactileSensor(Robot* robot, Body* body, string name, 
                    dtype kn = 5e5, dtype kt = 2.5e3, dtype mu = 1.5, dtype damping = 3e2,
                    bool render = false) 
                    : _robot(robot), _body(body), _name(name), 
                    _kn(kn), _kt(kt), _mu(mu), _damping(damping),
                    _render(render) {}

    void init(); // init class member variables

    void compute_tactile_values();   
    void compute_tactile_values_with_derivatives(MatrixX& dtactile_dqm, MatrixX& dtactile_dphi); // return shape: (num_markers * 3) x 12
    void test_derivatives_runtime();
    std::vector<Vector3> get_tactile_sensor_pos();
    void update_tactile_sensor_pos(std::vector<Vector3> new_pos_i);

    virtual void get_rendering_objects(
        std::vector<Matrix3Xf>& vertex_list, 
        std::vector<Matrix3Xi>& face_list,
        std::vector<opengl_viewer::Option>& option_list,
        std::vector<opengl_viewer::Animator*>& animator_list);

    Robot* _robot;
    Body* _body;
    string _name;
    bool _render;
    dtype _kn, _kt, _mu, _damping;

    std::vector<Vector3> _pos_i; // positions of each tactile sensor point in the associated body frame
    std::vector<Vector3> _normal_i; // normal direction of each tactile sensor point in the associated body frame
    std::vector<Vector3> _axis0_i, _axis1_i; // axis for visualizing the shear force
    std::vector<Vector2i> _image_pos; // positions of each tactile sensor point in the tactile sensor image
    std::vector<dtype> _depth;
    std::vector<dtype> _normal_force;
    std::vector<Vector2> _shear_force;
    std::vector<Vector3> _tactile_force;
    std::vector<Body*> _contact_body;

    // rendering
    Matrix3Xf _rendering_vertices;
    Matrix3Xi _rendering_faces;

protected:
    TactileSensorAnimator* _animator;
};

};