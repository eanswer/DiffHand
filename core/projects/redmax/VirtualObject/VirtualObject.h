#pragma once
#include "Common.h"
#include "Utils.h"

namespace redmax {

class Simulation;
class VirtualObject;

class VirtualObjectAnimator : public opengl_viewer::Animator {
public:
    VirtualObjectAnimator(VirtualObject* virtual_object) : _virtual_object(virtual_object) {}

    const Eigen::Matrix4f AnimatedModelMatrix(const float t);

    VirtualObject* _virtual_object;
};

class VirtualObject {
public:
    VirtualObject(const Simulation* sim, std::string name, int data_dim,
                    bool use_texture, Vector3 color, std::string texture_path);

    virtual void update_data(VectorX data) {}
    virtual VectorX get_data() = 0;
    int get_data_dim() { return _data_dim; }

    virtual Matrix4 get_transform_matrix() = 0;

    virtual void get_rendering_objects(
        std::vector<Matrix3Xf>& vertex_list, 
        std::vector<Matrix3Xi>& face_list,
        std::vector<opengl_viewer::Option>& option_list,
        std::vector<opengl_viewer::Animator*>& animator_list) {};

    const std::string _name;
    // simulation
    const Simulation* _sim;

    // color and texture
    Matrix3Xf _rendering_vertices;
    Matrix3Xi _rendering_faces;
    Vector3 _color;
    std::string _texture_path;
    bool _use_texture;
    int _data_dim;

protected:
    VirtualObjectAnimator* _animator;
};

}