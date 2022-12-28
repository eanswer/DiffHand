#pragma once
#include "Sensor/TactileSensor.h"

namespace redmax {

class TactileSensorRectArray : public TactileSensor {
public:
    TactileSensorRectArray(Robot* robot, Body* body, string name, 
                            Vector3 rect_pos0, Vector3 rect_pos1, 
                            Vector3 axis0, Vector3 axis1, Vector2i resolution, 
                            dtype kn = 5e5, dtype kt = 2.5e3, dtype mu = 1.5, dtype damping = 3e2,
                            bool render = false);

    void init();

    Vector3 _rect_pos0, _rect_pos1;
    Vector3 _axis0, _axis1;
    Vector2i _resolution;
};

};