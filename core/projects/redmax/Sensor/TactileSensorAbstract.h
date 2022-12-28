#pragma once
#include "Sensor/TactileSensor.h"

namespace redmax {

class TactileSensorAbstract : public TactileSensor {
public:
    TactileSensorAbstract(Robot* robot, Body* body, string name, string spec_filename, 
                    Matrix3 R_it, Vector3 p_it, 
                    dtype kn = 5e5, dtype kt = 2.5e3, dtype mu = 1.5, dtype damping = 3e2,
                    bool render = false);

    void init();

    std::string _spec_filename;
    Matrix3 _R_it;
    Vector3 _p_it;
};

};