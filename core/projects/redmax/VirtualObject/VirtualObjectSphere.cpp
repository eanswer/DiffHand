#include "VirtualObject/VirtualObjectSphere.h"
#include "Simulation.h"

namespace redmax {

VirtualObjectSphere::VirtualObjectSphere(const Simulation* sim, std::string name, Vector3 pos_world, dtype radius, 
                                        bool use_texture, Vector3 color, std::string texture_path) 
    : VirtualObject(sim, name, 6, use_texture, color, texture_path), _pos_world(pos_world), _radius(radius) {}

void VirtualObjectSphere::update_data(VectorX data) {
    if (data.size() != 3 && data.size() != 6) {
        throw_error("Update virtual sphere with wrong data size." );
    }

    if (data.size() == 3) {
        _pos_world = data.head(3);
    } else {
        _pos_world = data.head(3);
        _color = data.tail(3);
    }
}

VectorX VirtualObjectSphere::get_data() {
    VectorX data(6);
    data.head(3) = _pos_world;
    data.tail(3) = _color;

    return data;
}

Matrix4 VirtualObjectSphere::get_transform_matrix() {
    Matrix4 E = Matrix4::Identity();
    E.topRightCorner(3, 1) = _pos_world;
    return E;
}

// rendering
void VirtualObjectSphere::get_rendering_objects(
    std::vector<Matrix3Xf>& vertex_list, 
    std::vector<Matrix3Xi>& face_list,
    std::vector<opengl_viewer::Option>& option_list,
    std::vector<opengl_viewer::Animator*>& animator_list) {
    
    Matrix3Xf sphere_vertex;
    Matrix3Xi sphere_face;
    Matrix2Xf sphere_uv;
    opengl_viewer::Option object_option;

    opengl_viewer::ReadFromObjFile(
        std::string(GRAPHICS_CODEBASE_SOURCE_DIR) + "/resources/meshes/sphere.obj",
        sphere_vertex, sphere_face, sphere_uv);

    sphere_vertex *= (float)_radius;

    _rendering_vertices = sphere_vertex;
    _rendering_faces = sphere_face;
    
    if (_sim->_options->_unit == "cm-g") 
        sphere_vertex /= 10.;
    else
        sphere_vertex *= 10.;

    if (!_use_texture) {
        object_option.SetBoolOption("smooth normal", false);
        object_option.SetVectorOption("ambient", _color(0), _color(1), _color(2));
        object_option.SetVectorOption("diffuse", 0.4f, 0.2368f, 0.1036f);
        object_option.SetVectorOption("specular", 0.774597f, 0.458561f, 0.200621f);
        object_option.SetFloatOption("shininess", 76.8f);
    } else {
        opengl_viewer::Image checker_texture;
        checker_texture.Initialize(std::string(GRAPHICS_CODEBASE_SOURCE_DIR) + "//" + _texture_path);

        object_option.SetBoolOption("smooth normal", false);

        object_option.SetVectorOption("ambient", 0.7f, 0.7f, 0.7f);
        object_option.SetVectorOption("diffuse", 1.0f, 1.0f, 1.0f);
        object_option.SetVectorOption("specular", 1.0f, 1.0f, 1.0f);
        object_option.SetFloatOption("shininess", 1.5f);
        
        object_option.SetMatrixOption("uv", sphere_uv);
        object_option.SetMatrixOption("texture", checker_texture.rgb_data());
        object_option.SetIntOption("texture row num", checker_texture.row_num());
        object_option.SetIntOption("texture col num", checker_texture.col_num());
        object_option.SetStringOption("texture mag filter", "nearest");
    }

    _animator = new VirtualObjectAnimator(this);

    vertex_list.push_back(sphere_vertex);
    face_list.push_back(sphere_face);
    option_list.push_back(object_option);
    animator_list.push_back(_animator);
}

}