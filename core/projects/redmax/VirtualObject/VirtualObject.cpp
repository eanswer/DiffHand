#include "VirtualObject/VirtualObject.h"
#include "Simulation.h"

namespace redmax {
const Eigen::Matrix4f VirtualObjectAnimator::AnimatedModelMatrix(const float t) {
    Matrix4f model_matrix = (_virtual_object->get_transform_matrix()).cast<float>();
    if (_virtual_object->_sim->_options->_unit == "cm-g")
        model_matrix.topRightCorner(3, 1) /= 10.; // scale for better visualization
    else
        model_matrix.topRightCorner(3, 1) *= 10.; // scale for better visualization
    return model_matrix;
}

VirtualObject::VirtualObject(const Simulation* sim, std::string name, int data_dim,
                                bool use_texture, Vector3 color, std::string texture_path) 
    : _name(name), _sim(sim), _data_dim(data_dim), _use_texture(use_texture), _color(color), _texture_path(texture_path) {}

}