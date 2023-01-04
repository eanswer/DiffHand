#include "Sensor/TactileSensorRectArray.h"
#include "Body/Body.h"
#include "Body/BodyMeshObj.h"
#include "Body/BodyAbstract.h"

namespace redmax {

TactileSensorRectArray::TactileSensorRectArray(
    Robot* robot, Body* body, string name, Vector3 rect_pos0, Vector3 rect_pos1, 
    Vector3 axis0, Vector3 axis1, Vector2i resolution, 
    dtype kn, dtype kt, dtype mu, dtype damping,
    bool render) 
    : TactileSensor(robot, body, name, kn, kt, mu, damping, render) {
    
    _rect_pos0 = rect_pos0;
    _rect_pos1 = rect_pos1;
    _axis0 = axis0;
    _axis1 = axis1;
    _resolution = resolution;

    // in this case, current R_it is actually R_ot while object and body frames are not necessarily aligned.
    if (dynamic_cast<BodyMeshObj*>(const_cast<Body*>(body)) != nullptr) {
        Matrix4 _E_it = dynamic_cast<BodyMeshObj*>(const_cast<Body*>(body))->_E_io;
        Matrix3 _R_it = _E_it.topLeftCorner(3, 3);
        Vector3 _p_it = _E_it.topRightCorner(3, 1);
        _axis0 = _R_it * _axis0;
        _axis1 = _R_it * _axis1;
        _rect_pos0 = _R_it * _rect_pos0 + _p_it;
        _rect_pos1 = _R_it * _rect_pos1 + _p_it;
    } else if (dynamic_cast<BodyAbstract*>(const_cast<Body*>(body)) != nullptr) {
        Matrix4 _E_it = dynamic_cast<BodyAbstract*>(const_cast<Body*>(body))->_E_io;
        Matrix3 _R_it = _E_it.topLeftCorner(3, 3);
        Vector3 _p_it = _E_it.topRightCorner(3, 1);
        _axis0 = _R_it * _axis0;
        _axis1 = _R_it * _axis1;
        _rect_pos0 = _R_it * _rect_pos0 + _p_it;
        _rect_pos1 = _R_it * _rect_pos1 + _p_it;
    }

    init();
}

void TactileSensorRectArray::init() {
    // check whether input info is compatible
    dtype length_dir0 = (_rect_pos1 - _rect_pos0).dot(_axis0);
    dtype length_dir1 = (_rect_pos1 - _rect_pos0).dot(_axis1);
    if ((_rect_pos0 + length_dir0 * _axis0 + length_dir1 * _axis1 - _rect_pos1).norm() > 1e-5) {
        throw_error("Tactile info for " + _body->_name + " is incompatible");
    }

    // pre-compute the tactile sensor point locations
    Vector3 step_axis0 = length_dir0 / (_resolution(0) - 1) * _axis0;
    Vector3 step_axis1 = length_dir1 / (_resolution(1) - 1) * _axis1;

    Vector3 normal = _axis0.cross(_axis1);

    _pos_i.clear(); _normal_i.clear(); _axis0_i.clear(); _axis1_i.clear(); _image_pos.clear(); _depth.clear();
    for (int i = 0;i < _resolution(0);i++)
        for (int j = 0;j < _resolution(1);j++) {
            _pos_i.push_back(_rect_pos0 + step_axis0 * i + step_axis1 * j);
            _normal_i.push_back(normal);
            _axis0_i.push_back(_axis0);
            _axis1_i.push_back(_axis1);
            _image_pos.push_back(Vector2i(i, j));
        }
    
    TactileSensor::init();
}

};