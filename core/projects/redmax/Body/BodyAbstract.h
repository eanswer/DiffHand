#pragma once
#include "Body/Body.h"

namespace redmax {

/**
 * BodyAbstract is an abstract body.
 * The mass, inertia and contacts are directly specified instead of being inferred from object.
 * the rendering_mesh is only used for rendering but not related to any simulation dynamics
 * */
class BodyAbstract : public Body {
public:
    bool _rendering_mesh_exists;
    Matrix3X _V;            // vertices
    Matrix3Xi _F;            // face elements

    BodyAbstract(Simulation* sim, Joint* joint, 
        Matrix3 R_ji, Vector3 p_ji,
        dtype mass, Vector3 Inertia,
        std::string rendering_mesh_filename = "");
    
    void load_mesh(std::string filename);

    void set_rendering_mesh_vertices(const Matrix3X V);
    void set_rendering_mesh(const Matrix3X V, const Matrix3Xi F);
    
    // rendering
    void get_rendering_objects(
        std::vector<Matrix3Xf>& vertex_list, 
        std::vector<Matrix3Xi>& face_list,
        std::vector<opengl_viewer::Option>& option_list,
        std::vector<opengl_viewer::Animator*>& animator_list);
};

}