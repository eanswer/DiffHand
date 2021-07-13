#include "Body/Body.h"
#include "Body/BodyCuboid.h"
#include "Body/BodyMeshObj.h"
#include "Body/BodySphere.h"
#include "Body/BodyPrimitiveShape.h"
#include "CollisionDetection/CollisionDetection.h"
#include "ode/odeBoxBox.h"

namespace redmax {

// detect the collision between ground and a body
// return the list of the contact points.
void collision_detection_ground_object(const Matrix4 E_g, const Body* body, std::vector<Contact>& contacts) {
    Vector3 xg = E_g.topRightCorner(3, 1);
    Vector3 ng = E_g.block(0, 2, 3, 1);

    if (dynamic_cast<BodySphere*>(const_cast<Body*>(body)) != nullptr) { // special collision detection for sphere
        Vector3 center_world = body->_E_0i.topRightCorner(3, 1);
        dtype r = dynamic_cast<BodySphere*>(const_cast<Body*>(body))->_radius;
        dtype d = ng.dot(center_world - xg) - r;
        if (d <= 0.) {
            Vector3 xw = center_world - ng * r;
            Vector3 xi = body->_E_i0.topLeftCorner(3, 3) * xw + body->_E_i0.topRightCorner(3, 1);
            contacts.push_back(Contact(xi, xw, d, ng));
        }
    } else {
        std::vector<Vector3> contact_points = body->get_contact_points();
        MatrixX xi(4, contact_points.size());
        for (int i = 0;i < contact_points.size();i++) {
            xi.col(i) = Vector4(contact_points[i][0], contact_points[i][1], contact_points[i][2], 1);
        }

        MatrixX xw = body->_E_0i * xi;

        for (int i = 0;i < contact_points.size();i++) {
            dtype d = ng.dot(xw.col(i).head(3) - xg);
            if (d <= 0.) {
                contacts.push_back(Contact(xi.col(i).head(3), xw.col(i).head(3), d, ng, i));
            }
        }
    } 
}

// detect the collision points between two cuboid bodies
// return the list of the contact points.
// use odeBoxBox
void collision_detection_cuboid_cuboid(const BodyCuboid* cuboid1, const BodyCuboid* cuboid2, std::vector<Contact>& contacts) {
    Eigen::Matrix4d E1 = cuboid1->_E_0i.cast<double>();
    Eigen::Matrix4d E2 = cuboid2->_E_0i.cast<double>();
    Eigen::Vector3d s1 = cuboid1->_length.cast<double>();
    Eigen::Vector3d s2 = cuboid2->_length.cast<double>();

    ode::Contacts results = ode::odeBoxBox(E1, s1, E2, s2);

    for (int i = 0;i < results.count;i++) {
        contacts.push_back(Contact(Vector3::Zero(), 
            results.positions[i].cast<dtype>(),
            (dtype)results.depths[i],
            results.normal.cast<dtype>()));
    }
}

// detect the collision between a general body and a primitive body
// the general contact body should be able to give a list of contact points on the surface.
// the primitive body is supposed to have an anlytical distance field
void collision_detection_general_primitive(const Body* contact_body, const Body* primitive_body, std::vector<Contact>& contacts) {
    if (dynamic_cast<BodyPrimitiveShape*>(const_cast<Body*>(primitive_body)) != nullptr) {
        BodyPrimitiveShape* primitive_body_ = dynamic_cast<BodyPrimitiveShape*>(const_cast<Body*>(primitive_body));
        std::vector<Vector3> contact_points = contact_body->get_contact_points();
        
        if (dynamic_cast<BodyCuboid*>(const_cast<Body*>(contact_body)) != nullptr) {
            BodyCuboid* cuboid_body_ = dynamic_cast<BodyCuboid*>(const_cast<Body*>(contact_body));
            contact_points = cuboid_body_->get_general_contact_points();
        }

        Matrix4 E_w1 = contact_body->_E_0i;
        
        for (int i = 0;i < contact_points.size();i++) {
            Vector3 xw1 = E_w1.topLeftCorner(3, 3) * contact_points[i] + E_w1.topRightCorner(3, 1);
            dtype d = primitive_body_->distance(xw1);
            if (d < 0.) {
                contacts.push_back(Contact(contact_points[i], xw1, d, Vector3::Zero(), i));
            }    
        }
    } else {
        // not implemented yet
    }
}

}