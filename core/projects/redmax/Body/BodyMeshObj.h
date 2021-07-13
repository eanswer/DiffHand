#pragma once
#include "Body/Body.h"

namespace redmax {

/**
 * BodyMeshObj define a body loaded from an .obj file (triangle surface mesh).
 **/
class BodyMeshObj : public Body {
public:
    std::string _filename;  // .obj filename (path)
    Matrix3X _V;            // vertices
    Matrix3Xi _F;            // face elements
    SE3 _E_oi;               // transform from body frame to obj frame
    SE3 _E_io;               // transform from obj frame to body frame
    SE3 _E_0i;               // transform from body frame to world frame

    enum TransformType {
        BODY_TO_JOINT,
        OBJ_TO_WOLRD,
        OBJ_TO_JOINT
    };

    BodyMeshObj(Simulation* sim, Joint* joint,
                    std::string filename, 
                    Matrix3 R, Vector3 p, 
                    TransformType transform_type = BODY_TO_JOINT,
                    dtype density = (dtype)1.0);

    // rendering
    void get_rendering_objects(
        std::vector<Matrix3Xf>& vertex_list, 
        std::vector<Matrix3Xi>& face_list,
        std::vector<opengl_viewer::Option>& option_list,
        std::vector<opengl_viewer::Animator*>& animator_list);

private:
    void load_mesh(std::string filename);

    void process_mesh();

    void compute_mass_property(const Matrix3X &V, const Matrix3Xi &F, /*input*/
                                dtype &mass, Vector3 &COM,          /*output*/
                                Matrix3 &I);

    void precompute_contact_points();
};

}