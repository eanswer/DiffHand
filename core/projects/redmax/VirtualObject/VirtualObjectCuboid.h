#pragma once
#include "Common.h"
#include "Utils.h"
#include "VirtualObject/VirtualObject.h"

namespace redmax {

class VirtualObjectCuboid : public VirtualObject {
public:
    VirtualObjectCuboid(const Simulation* sim, std::string name, 
                        Vector3 length, Vector3 pos_world, Matrix3 R,
                        bool use_texture, Vector3 color, std::string texture_path);

    // data can be either in size 7 (xyz location, quaternion) or in length 10 (xyz location, quaternion, rgb color)
    void update_data(VectorX data);
    VectorX get_data();
    
    // rendering
    void get_rendering_objects(
        std::vector<Matrix3Xf>& vertex_list, 
        std::vector<Matrix3Xi>& face_list,
        std::vector<opengl_viewer::Option>& option_list,
        std::vector<opengl_viewer::Animator*>& animator_list);

    Matrix4 get_transform_matrix();
    
private:
    Vector3 _length;
    Vector3 _pos_world;
    Matrix3 _R;
};

}