#include "VirtualObject/VirtualObjectCuboid.h"
#include "Simulation.h"

namespace redmax {

VirtualObjectCuboid::VirtualObjectCuboid(
    const Simulation* sim, std::string name, 
    Vector3 length, Vector3 pos_world, Matrix3 R, bool use_texture, 
    Vector3 color, std::string texture_path)
    : VirtualObject(sim, name, 10, use_texture, color, texture_path), 
                    _length(length), _pos_world(pos_world), _R(R) {}

void VirtualObjectCuboid::update_data(VectorX data) {
    if (data.size() != 7 && data.size() != 10) {
        throw_error("Update virtual cuboid with wrong data size." );
    }

    if (data.size() == 7) {
        _pos_world = data.head(3);
        Vector4 quat = data.tail(4);
        _R = math::quat2mat(quat);
    } else {
        _pos_world = data.head(3);
        Vector4 quat = data.segment(3, 4);
        _R = math::quat2mat(quat);
        _color = data.tail(3);
    }
}

VectorX VirtualObjectCuboid::get_data() {
    VectorX data(10);
    data.head(3) = _pos_world;
    data.segment(3, 4) = math::mat2quat(_R);
    Matrix3 R_tmp = math::quat2mat(data.segment(3, 4));
    dtype error = (R_tmp - _R).norm();
    
    data.tail(3) = _color;

    return data;
}

Matrix4 VirtualObjectCuboid::get_transform_matrix() {
    Matrix4 E = Matrix4::Identity();
    E.topLeftCorner(3, 3) = _R;
    E.topRightCorner(3, 1) = _pos_world;
    return E;
}

// rendering
void VirtualObjectCuboid::get_rendering_objects(
    std::vector<Matrix3Xf>& vertex_list, 
    std::vector<Matrix3Xi>& face_list,
    std::vector<opengl_viewer::Option>& option_list,
    std::vector<opengl_viewer::Animator*>& animator_list) {
    
    Matrix3Xf cube_vertex;
    Matrix3Xi cube_face;
    Matrix2Xf cube_uv;
    opengl_viewer::Option object_option;

    opengl_viewer::ReadFromObjFile(
        std::string(GRAPHICS_CODEBASE_SOURCE_DIR) + "/resources/meshes/cube.obj",
        cube_vertex, cube_face, cube_uv);

    // set cube uv
    Eigen::Vector2f texture_bottom_left;
    Eigen::Vector2f texture_top_right;
    Eigen::Vector2f texture_pos;
    for (int i = 0;i < cube_vertex.cols();i++) {
        if (cube_vertex(0, i) <= -0.5 + math::eps) { // middle (1)
            texture_bottom_left = Eigen::Vector2f(0.25, 2. / 3.);
            texture_top_right = Eigen::Vector2f(0., 1. / 3.);
            texture_pos = Eigen::Vector2f(cube_vertex(1, i), cube_vertex(2, i)) + Eigen::Vector2f(0.5, 0.5);
        } else if (cube_vertex(1, i) <= -0.5 + math::eps) { // middle (2)
            texture_bottom_left = Eigen::Vector2f(0.25, 2. / 3.);
            texture_top_right = Eigen::Vector2f(0.5, 1. / 3.);
            texture_pos = Eigen::Vector2f(cube_vertex(0, i), cube_vertex(2, i)) + Eigen::Vector2f(0.5, 0.5);
        } else if (cube_vertex(0, i) >= 0.5 - math::eps) { // middle (3)
            texture_bottom_left = Eigen::Vector2f(0.5, 2. / 3.);
            texture_top_right = Eigen::Vector2f(0.75, 1. / 3.);
            texture_pos = Eigen::Vector2f(cube_vertex(1, i), cube_vertex(2, i)) + Eigen::Vector2f(0.5, 0.5);
        } else if (cube_vertex(1, i) >= 0.5 - math::eps) { // middle (4)
            texture_bottom_left = Eigen::Vector2f(1., 2. / 3.);
            texture_top_right = Eigen::Vector2f(0.75, 1. / 3.);
            texture_pos = Eigen::Vector2f(cube_vertex(0, i), cube_vertex(2, i)) + Eigen::Vector2f(0.5, 0.5);
        } else if (cube_vertex(2, i) >= 0.5 - math::eps) { // top
            texture_bottom_left = Eigen::Vector2f(0.25, 1. / 3.);
            texture_top_right = Eigen::Vector2f(0.5 - 0.01, 0. + 0.01);
            texture_pos = Eigen::Vector2f(cube_vertex(0, i), cube_vertex(1, i)) + Eigen::Vector2f(0.5, 0.5);
        } else { // bottom
            texture_bottom_left = Eigen::Vector2f(0.25, 2. / 3.);
            texture_top_right = Eigen::Vector2f(0.5 - 0.01, 1. - 0.01);
            texture_pos = Eigen::Vector2f(cube_vertex(0, i), cube_vertex(1, i)) + Eigen::Vector2f(0.5, 0.5);
        }
        cube_uv.col(i) = texture_bottom_left + (texture_top_right - texture_bottom_left).cwiseProduct(texture_pos);
    }

    for (int i = 0;i < 3;i++)
        cube_vertex.row(i) *= (float)_length(i);

    _rendering_vertices = cube_vertex;
    _rendering_faces = cube_face;
    
    if (_sim->_options->_unit == "cm-g") 
        cube_vertex /= 10.;
    else
        cube_vertex *= 10.;

    if (!_use_texture) {
        object_option.SetVectorOption("ambient", _color(0), _color(1), _color(2));
        object_option.SetVectorOption("diffuse", 0.4f, 0.2368f, 0.1036f);
        object_option.SetVectorOption("specular", 0.474597f, 0.358561f, 0.200621f);
        object_option.SetFloatOption("shininess", 46.8f);
    } else {
        opengl_viewer::Image checker_texture;
        checker_texture.Initialize(std::string(GRAPHICS_CODEBASE_SOURCE_DIR) + "//" + _texture_path);

        object_option.SetBoolOption("smooth normal", false);

        object_option.SetVectorOption("ambient", 0.7f, 0.7f, 0.7f);
        object_option.SetVectorOption("diffuse", 1.0f, 1.0f, 1.0f);
        object_option.SetVectorOption("specular", 1.0f, 1.0f, 1.0f);
        object_option.SetFloatOption("shininess", 1.5f);
        
        object_option.SetMatrixOption("uv", cube_uv);
        object_option.SetMatrixOption("texture", checker_texture.rgb_data());
        object_option.SetIntOption("texture row num", checker_texture.row_num());
        object_option.SetIntOption("texture col num", checker_texture.col_num());
        object_option.SetStringOption("texture mag filter", "nearest");
    }

    _animator = new VirtualObjectAnimator(this);

    vertex_list.push_back(cube_vertex);
    face_list.push_back(cube_face);
    option_list.push_back(object_option);
    animator_list.push_back(_animator);
}

}