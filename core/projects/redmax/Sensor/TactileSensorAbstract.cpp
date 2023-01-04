#include "Sensor/TactileSensorAbstract.h"
#include "Body/Body.h"
#include "Body/BodyMeshObj.h"
#include "Body/BodyAbstract.h"

namespace redmax {

TactileSensorAbstract::TactileSensorAbstract(
    Robot* robot, Body* body, string name, string spec_filename,
    Matrix3 R_it, Vector3 p_it, 
    dtype kn, dtype kt, dtype mu, dtype damping,
    bool render) 
    : TactileSensor(robot, body, name, kn, kt, mu, damping, render) {
    
    _spec_filename = spec_filename;
    _R_it = R_it;
    _p_it = p_it;

    // in this case, current R_it is actually R_ot while object and body frames are not necessarily aligned.
    if (dynamic_cast<BodyMeshObj*>(const_cast<Body*>(body)) != nullptr) { 
        Matrix4 _E_ot = math::SE(_R_it, _p_it);
        Matrix4 _E_it = dynamic_cast<BodyMeshObj*>(const_cast<Body*>(body))->_E_io * _E_ot;
        _R_it = _E_it.topLeftCorner(3, 3);
        _p_it = _E_it.topRightCorner(3, 1);
    } else if (dynamic_cast<BodyAbstract*>(const_cast<Body*>(body)) != nullptr) {
        Matrix4 _E_ot = math::SE(_R_it, _p_it);
        Matrix4 _E_it = dynamic_cast<BodyAbstract*>(const_cast<Body*>(body))->_E_io * _E_ot;
        _R_it = _E_it.topLeftCorner(3, 3);
        _p_it = _E_it.topRightCorner(3, 1);
    }

    init();
}

void TactileSensorAbstract::init() {
    FILE* fp = fopen(_spec_filename.c_str(), "r");
    
    int N;
    fscanf(fp, "%d", &N);
    _pos_i.clear(); _normal_i.clear(); _axis0_i.clear(); _axis1_i.clear(); _image_pos.clear(); _depth.clear();
    for (int i = 0;i < N;++i) {
        char ch;
        std::string st;

        // sensor pos
        for (;;) {
            ch = fgetc(fp);
            if (ch == '\"') {
                break;
            }
        }
        st = "";
        for (;(ch = fgetc(fp)) != '\"';) {
            st = st + ch;
        }

        Vector3 sensor_pos = str_to_eigen(st);

        // image pos
        for (;;) {
            ch = fgetc(fp);
            if (ch == '\"') {
                break;
            }
        }
        st = "";
        for (;(ch = fgetc(fp)) != '\"';) {
            st = st + ch;
        }

        Vector2i image_pos = str_to_eigen_int(st);

        // normal
        for (;;) {
            ch = fgetc(fp);
            if (ch == '\"') {
                break;
            }
        }
        st = "";
        for (;(ch = fgetc(fp)) != '\"';) {
            st = st + ch;
        }

        Vector3 normal = str_to_eigen(st);

        // axis 0
        for (;;) {
            ch = fgetc(fp);
            if (ch == '\"') {
                break;
            }
        }
        st = "";
        for (;(ch = fgetc(fp)) != '\"';) {
            st = st + ch;
        }

        Vector3 axis_0 = str_to_eigen(st);

        // axis 1
        for (;;) {
            ch = fgetc(fp);
            if (ch == '\"') {
                break;
            }
        }
        st = "";
        for (;(ch = fgetc(fp)) != '\"';) {
            st = st + ch;
        }

        Vector3 axis_1 = str_to_eigen(st);

        _pos_i.push_back(_R_it * sensor_pos + _p_it);
        _image_pos.push_back(image_pos);
        _normal_i.push_back(_R_it * normal + _p_it);
        _axis0_i.push_back(_R_it * axis_0 + _p_it);
        _axis1_i.push_back(_R_it * axis_1 + _p_it);
    }
    fclose(fp);

    TactileSensor::init();
}

};