#include "Body/BodyAbstract.h"
#include "tiny_obj_loader.h"

namespace redmax {

BodyAbstract::BodyAbstract(
    Simulation* sim, Joint* joint, 
    Matrix3 R_ji, Vector3 p_ji,
    dtype mass, Vector3 Inertia,
    std::string rendering_mesh_filename)
    : Body(sim, joint, R_ji, p_ji, 0.0) {

    if (rendering_mesh_filename != "") {
        _rendering_mesh_exists = true;
        load_mesh(rendering_mesh_filename);
    } else {
        _rendering_mesh_exists = false;
    }
    
    _mass = mass;
    _Inertia.head(3) = Inertia;
    _Inertia.tail(3).setConstant(_mass);
}

void BodyAbstract::load_mesh(std::string filename) {
    std::vector<tinyobj::shape_t> obj_shape;
    std::vector<tinyobj::material_t> obj_material;
    tinyobj::attrib_t attrib;
    std::string err;
    tinyobj::LoadObj(&attrib, &obj_shape, &obj_material, &err, filename.c_str());

    int num_vertices = (int)attrib.vertices.size() / 3;
    _V.resize(3, num_vertices);
    for (int i = 0;i < num_vertices;i++) {
        _V.col(i) = Vector3(attrib.vertices[i * 3], 
            attrib.vertices[i * 3 + 1],
            attrib.vertices[i * 3 + 2]);
    }
    
    int num_elements = (int)obj_shape[0].mesh.indices.size() / 3;
    _F.resize(3, num_elements);
    for (int i = 0;i < num_elements;i++) {
        _F.col(i) = Vector3i(obj_shape[0].mesh.indices[i * 3].vertex_index,
            obj_shape[0].mesh.indices[i * 3 + 1].vertex_index,
            obj_shape[0].mesh.indices[i * 3 + 2].vertex_index);
    }
}

void BodyAbstract::set_rendering_mesh_vertices(const Matrix3X V) {
    _V = V;
}

void BodyAbstract::set_rendering_mesh(const Matrix3X V, const Matrix3Xi F) {
    _V = V;
    _F = F;
}
    
void BodyAbstract::get_rendering_objects(
    std::vector<Matrix3Xf>& vertex_list, 
    std::vector<Matrix3Xi>& face_list,
    std::vector<opengl_viewer::Option>& option_list,
    std::vector<opengl_viewer::Animator*>& animator_list) {

    _animator = new BodyAnimator(this);

    if (_rendering_mesh_exists) {
        opengl_viewer::Option object_option;

        object_option.SetBoolOption("smooth normal", false);
        object_option.SetVectorOption("ambient", _color(0), _color(1), _color(2));
        object_option.SetVectorOption("diffuse", 0.4f, 0.2368f, 0.1036f);
        object_option.SetVectorOption("specular", 0.774597f, 0.658561f, 0.400621f);
        object_option.SetFloatOption("shininess", 76.8f);

        Matrix3Xf vertex = _V.cast<float>() / 10.;

        vertex_list.push_back(vertex);
        face_list.push_back(_F);
        option_list.push_back(object_option);
    } else {
        opengl_viewer::Option object_option;

        object_option.SetVectorOption("ambient", _color(0), _color(1), _color(2));
        object_option.SetVectorOption("diffuse", 0.4f, 0.2368f, 0.1036f);
        object_option.SetVectorOption("specular", 0.474597f, 0.358561f, 0.200621f);
        object_option.SetFloatOption("shininess", 46.8f);

        Matrix3Xf cube_vertex;
        Matrix3Xi cube_face;
        Matrix2Xf cube_uv;

        opengl_viewer::ReadFromObjFile(
            std::string(GRAPHICS_CODEBASE_SOURCE_DIR) + "/resources/meshes/cube.obj",
            cube_vertex, cube_face, cube_uv);
        
        Vector3 length;
        length(0) = sqrt((_Inertia(2) + _Inertia(1) - _Inertia(0)) * 6 / _mass);
        length(1) = sqrt((_Inertia(2) + _Inertia(0) - _Inertia(1)) * 6 / _mass);
        length(2) = sqrt((_Inertia(1) + _Inertia(0) - _Inertia(2)) * 6 / _mass);
        
        for (int i = 0;i < 3;i++)
            cube_vertex.row(i) *= (float)length(i) / 10.;
        
        vertex_list.push_back(cube_vertex);
        face_list.push_back(cube_face);
        option_list.push_back(object_option);
    }

    animator_list.push_back(_animator);
}

}