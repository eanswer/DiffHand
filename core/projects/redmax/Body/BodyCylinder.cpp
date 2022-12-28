#include "BodyCylinder.h"
#include "Simulation.h"

namespace redmax {

BodyCylinder::BodyCylinder(Simulation *sim, Joint *joint, 
                dtype radius, dtype length,
                Matrix3 R_ji, Vector3 p_ji, dtype density,
                int general_contact_angle_resolution,
                int general_contact_radius_resolution) 
            : BodyPrimitiveShape(sim, joint, R_ji, p_ji, density) {
    _radius = radius;
    _length = length;
    _general_contact_angle_resolution = general_contact_angle_resolution;
    _general_contact_radius_resolution = general_contact_radius_resolution;

    precompute_contact_points();
    computeMassMatrix();
}

void BodyCylinder::computeMassMatrix() {
    _mass = constants::pi * _radius * _radius * _length * _density;
    _Inertia(0) = _mass * _length * _length / 12. + _mass * _radius * _radius / 4.;
    _Inertia(1) = _mass * _length * _length / 12. + _mass * _radius * _radius / 4.;
    _Inertia(2) = _mass * _radius * _radius / 2.;
    _Inertia.tail(3).noalias() = Vector3::Constant(_mass);
}

// we only consider the contat points on two side faces for now
void BodyCylinder::precompute_contact_points() {
    _contact_points.clear();
    for (int dz = -1;dz <= 1;dz += 2) { // two faces
        _contact_points.push_back(Vector3(0., 0., dz * _length / 2.));
        for (int angle_idx = 0;angle_idx < _general_contact_angle_resolution;++angle_idx) {
            dtype angle = constants::pi * 2. / _general_contact_angle_resolution * angle_idx;
            for (int radius_idx = 0;radius_idx < _general_contact_radius_resolution;++radius_idx) {
                Vector3 point = Vector3(0., 0., dz * _length / 2.) 
                                + Vector3(cos(angle) * (radius_idx + 1) / _general_contact_radius_resolution * _radius, 
                                          sin(angle) * (radius_idx + 1) / _general_contact_radius_resolution * _radius,
                                          0.);
                _contact_points.push_back(point);
            }
        }
    }
}

// rendering
void BodyCylinder::get_rendering_objects(
    std::vector<Matrix3Xf>& vertex_list, 
    std::vector<Matrix3Xi>& face_list,
    std::vector<opengl_viewer::Option>& option_list,
    std::vector<opengl_viewer::Animator*>& animator_list) {

    Matrix3Xf cube_vertex;
    Matrix3Xi cube_face;
    Matrix2Xf cube_uv;
    opengl_viewer::Option object_option;

    opengl_viewer::ReadFromObjFile(
        std::string(GRAPHICS_CODEBASE_SOURCE_DIR) + "/resources/meshes/cylinder.obj",
        cube_vertex, cube_face, cube_uv);

    cube_vertex.row(0) *= (float)_radius;
    cube_vertex.row(1) *= (float)_radius;
    cube_vertex.row(2) *= (float)_length;

    _rendering_vertices = cube_vertex;
    _rendering_faces = cube_face;
    
    if (_sim->_options->_unit == "cm-g") 
        cube_vertex /= 10.;
    else
        cube_vertex *= 10.;

    object_option.SetVectorOption("ambient", _color(0), _color(1), _color(2));
    object_option.SetVectorOption("diffuse", 0.4f, 0.2368f, 0.1036f);
    object_option.SetVectorOption("specular", 0.474597f, 0.358561f, 0.200621f);
    object_option.SetFloatOption("shininess", 46.8f);

    _animator = new BodyAnimator(this);

    vertex_list.push_back(cube_vertex);
    face_list.push_back(cube_face);
    option_list.push_back(object_option);
    animator_list.push_back(_animator);
}

// TODO: no contact to the top/bottom surfaces
dtype BodyCylinder::distance(Vector3 xw) {
    Vector3 x = _E_i0.topLeftCorner(3, 3) * xw + _E_i0.topRightCorner(3, 1);
    dtype s = _length / 2.;
    dtype d;
    if (x(2) < -s) {
        d = 1.;
    } else if (x(2) > s) {
        d = 1.;
    } else {
        d = sqrt(x(0) * x(0) + x(1) * x(1)) - _radius;
    }
    return d;
}

// TODO: no contact to the top/bottom surfaces
void BodyCylinder::collision(
    Vector3 xw, Vector3 xw_dot, /* input */
    dtype &d, Vector3 &n,  /* output */
    dtype &ddot, Vector3 &tdot,
    Vector3 &xi2) {

    Matrix3 I = Matrix3::Identity();
    Matrix3 R2 = _E_0i.topLeftCorner(3, 3);
    Vector3 p2 = _E_0i.topRightCorner(3, 1);
    Vector3 w2_dot = _phi.head(3);
    Vector3 v2_dot = _phi.tail(3);

    dtype s = _length / 2.;

    Vector3 x = R2.transpose() * (xw - p2);
    Vector3 xdot = math::skew(w2_dot).transpose() * x + R2.transpose() * xw_dot - v2_dot;

    Vector3 center;
    if (x(2) < -s) {
        return;
    } else if (x(2) > s) {
        return;
    } else {
        center = Vector3(0., 0., x(2));
    }

    d = (x - center).norm() - _radius;
    Vector3 n2 = (x - center) / (x - center).norm();
    n = R2 * n2;
    ddot = n2.transpose() * xdot;
    xi2 = x - d * n2;

    Vector3 xw2_dot = R2 * (w2_dot.cross(xi2) + v2_dot);
    Vector3 vw = xw_dot - xw2_dot;
    tdot = (I - n * n.transpose()) * vw;
}

// TODO
void BodyCylinder::collision(
    Vector3 xw, Vector3 xw_dot, /* input */
    dtype &d, Vector3 &n,  /* output */
    dtype &ddot, Vector3 &tdot,
    Vector3 &xi2,
    RowVector3 &dd_dxw, RowVector6 &dd_dq2, /* derivatives for d */ 
    Matrix3 &dn_dxw, Matrix36 &dn_dq2, /* derivatives for n */
    RowVector3 &dddot_dxw, RowVector3 &dddot_dxwdot, /* derivatives for ddot */
    RowVector6 &dddot_dq2, RowVector6 &dddot_dphi2,
    Matrix3 &dtdot_dxw, Matrix3 &dtdot_dxwdot, /* derivatives for tdot */
    Matrix36 &dtdot_dq2, Matrix36 &dtdot_dphi2,
    Matrix3 &dxi2_dxw, Matrix36 &dxi2_dq2/* derivatives for xi2 */) {
    
    Matrix3 I = Matrix3::Identity();
    Matrix3 R2 = _E_0i.topLeftCorner(3, 3);
    Vector3 p2 = _E_0i.topRightCorner(3, 1);
    Vector3 w2_dot = _phi.head(3);
    Vector3 v2_dot = _phi.tail(3);

    dtype s = _length / 2.;

    /**************** values ****************/
    Vector3 x = R2.transpose() * (xw - p2);
    Vector3 xdot = math::skew(w2_dot).transpose() * x + R2.transpose() * xw_dot - v2_dot;

    Vector3 center;
    if (x(2) < -s) {
        return;
    } else if (x(2) > s) {
        return;
    } else {
        center = Vector3(0., 0., x(2));
    }

    Vector3 r = x - center;
    dtype r_norm = r.norm();
    d = r_norm - _radius;
    Vector3 n2 = r / r_norm;
    n = R2 * n2;
    RowVectorX dd_dx = n2.transpose();
    ddot = dd_dx * xdot;
    xi2 = x - d * n2;

    Vector3 xw2_dot = R2 * (w2_dot.cross(xi2) + v2_dot);
    Vector3 vw = xw_dot - xw2_dot;
    tdot = (I - n * n.transpose()) * vw;

    /**************** derivatives ****************/

    // r
    Matrix3 dr_dx;
    if (x(2) < -s || x(2) > s) {
        dr_dx = I;
    } else {
        dr_dx = I;
        dr_dx(2, 2) = 0.;
    }

    // x
    Matrix3 dx_dxw = R2.transpose();
    Matrix3 dx_dw2 = math::skew(x);
    Matrix3 dx_dv2 = -I;

    // d
    dd_dxw = dd_dx * dx_dxw;
    dd_dq2.head(3) = dd_dx * dx_dw2;
    dd_dq2.tail(3) = dd_dx * dx_dv2;

    // n2, n2 = r / r_norm
    Matrix3 dn2_dx = dr_dx / r_norm - r * r.transpose() * dr_dx / (r_norm * r_norm * r_norm);

    Matrix3 dn2_dxw = dn2_dx * dx_dxw;
    Matrix3 dn2_dw2 = dn2_dx * dx_dw2;
    Matrix3 dn2_dv2 = dn2_dx * dx_dv2;

    // n, n = R2 * n2
    Matrix3 dn_dx = R2 * dn2_dx;

    dn_dxw = dn_dx * dx_dxw;
    dn_dq2.leftCols(3) = -R2 * math::skew(n2) + dn_dx * dx_dw2;
    dn_dq2.rightCols(3) = dn_dx * dx_dv2;

    // xi2, xi2 = x - d * n2
    dxi2_dxw = dx_dxw - n2 * dd_dxw - d * dn2_dxw;
    dxi2_dq2.leftCols(3) = dx_dw2 - n2 * dd_dq2.leftCols(3) - d * dn2_dw2;
    dxi2_dq2.rightCols(3) = dx_dv2 - n2 * dd_dq2.rightCols(3) - d * dn2_dv2;

    // xdot, xdot = [w2_dot]'*x + R2'*xw_dot - v2_dot
    Matrix3 dxdot_dxw = math::skew(w2_dot).transpose() * R2.transpose();
    Matrix3 dxdot_dxwdot = R2.transpose();
    Matrix3 dxdot_dw2 = math::skew(w2_dot).transpose() * dx_dw2 + math::skew(R2.transpose() * xw_dot);
    Matrix3 dxdot_dv2 = -math::skew(w2_dot).transpose();
    Matrix3 dxdot_dw2dot;
    dxdot_dw2dot.col(0) = math::skew(Vector3::UnitX()).transpose() * x;
    dxdot_dw2dot.col(1) = math::skew(Vector3::UnitY()).transpose() * x;
    dxdot_dw2dot.col(2) = math::skew(Vector3::UnitZ()).transpose() * x;
    Matrix3 dxdot_dv2dot = -I;

    // ddot, ddot = n2' * xdot
    dddot_dxw = xdot.transpose() * dn2_dxw + n2.transpose() * dxdot_dxw;
    dddot_dxwdot = n2.transpose() * dxdot_dxwdot;
    dddot_dq2.head(3) = xdot.transpose() * dn2_dw2 + n2.transpose() * dxdot_dw2;
    dddot_dq2.tail(3) = xdot.transpose() * dn2_dv2 + n2.transpose() * dxdot_dv2;
    dddot_dphi2.head(3) = n2.transpose() * dxdot_dw2dot;
    dddot_dphi2.tail(3) = n2.transpose() * dxdot_dv2dot;

    // xw2_dot, xw2_dot = R2 * (w2_dot.cross(xi2) + v2_dot)
    Matrix3 dxw2dot_dxw = R2 * math::skew(w2_dot) * dxi2_dxw;
    Matrix3 dxw2dot_dxwdot = Matrix3::Zero();
    Matrix36 dxw2dot_dq2;
    dxw2dot_dq2.leftCols(3) = -R2 * math::skew(w2_dot.cross(xi2) + v2_dot) + R2 * math::skew(w2_dot) * dxi2_dq2.leftCols(3);
    dxw2dot_dq2.rightCols(3) = R2 * math::skew(w2_dot) * dxi2_dq2.rightCols(3);
    Matrix36 dxw2dot_dphi2;
    Vector3 e1 = Vector3::UnitX(), e2 = Vector3::UnitY(), e3 = Vector3::UnitZ();
    dxw2dot_dphi2.col(0) = R2 * math::skew(e1) * xi2;
    dxw2dot_dphi2.col(1) = R2 * math::skew(e2) * xi2;
    dxw2dot_dphi2.col(2) = R2 * math::skew(e3) * xi2;
    dxw2dot_dphi2.rightCols(3) = R2;

    // tdot, tdot = (I - n * n') * (xw_dot - xw2_dot)
    dtdot_dxw = -(I - n * n.transpose()) * dxw2dot_dxw - 
                    (n.transpose() * vw * dn_dxw
                    + n * vw.transpose() * dn_dxw);
    dtdot_dxwdot = I - n * n.transpose();
    dtdot_dq2 = -(I - n * n.transpose()) * dxw2dot_dq2 - 
                    (n.transpose() * vw * dn_dq2
                    + n * vw.transpose() * dn_dq2);
    dtdot_dphi2 = (I - n * n.transpose()) * (- dxw2dot_dphi2);
}

void BodyCylinder::update_density(dtype density) {
    _density = density;
    computeMassMatrix();
}

void BodyCylinder::update_size(VectorX body_size) {
    assert(body_size.size() == 2);
    _length = body_size(0);
    _radius = body_size(1);
    precompute_contact_points();
    computeMassMatrix();
}

void BodyCylinder::test_collision_derivatives() {
    // srand(1000);
    srand(time(0));
    for (int ii = 0;ii < 200;++ii) {
        dtype eps = 1e-8;
        
        // generate random xw, xw_dot, E_2, phi_2
        Eigen::Quaternion<dtype> quat_2(Vector4::Random());
        quat_2.normalize();
        Matrix4 E2 = Matrix4::Identity();
        E2.topLeftCorner(3, 3) = quat_2.toRotationMatrix();
        E2.topRightCorner(3, 1) = Vector3::Random() * 10.;
        Vector6 phi2 = Vector6::Random() * 10.;
        
        _length = 1.0;
        _radius = 0.3;
        dtype s = _length / 2.;

        Vector3 x = Vector3::Random();
        x(0) = x(0) * _radius * 2.;
        x(1) = x(1) * _radius * 2.;
        x(2) = -s + (x(2) + 1.) * s;
        
        Vector3 xw1 = E2.topLeftCorner(3, 3) * x + E2.topRightCorner(3, 1);
        Vector3 xw1_dot = Vector3::Random() * 10.;

        // std::cerr << "x = " << x.transpose() << std::endl;
        // std::cerr << "xw1 = " << xw1.transpose() << std::endl;
        // std::cerr << "xw1_dot = " << xw1_dot.transpose() << std::endl;
        // std::cerr << "E2 = " << std::endl << E2 << std::endl;
        // std::cerr << "phi2 = " << phi2.transpose() << std::endl;

        this->_E_0i = E2;
        this->_phi = phi2;
        dtype d, ddot;
        Vector3 n, tdot, xi2;
        RowVector3 dd_dxw1, dddot_dxw1, dddot_dxw1dot;
        RowVector6 dd_dq2, dddot_dq2, dddot_dphi2;
        Matrix3 dn_dxw1, dtdot_dxw1, dtdot_dxw1dot, dxi2_dxw1;
        Matrix36 dn_dq2, dtdot_dq2, dtdot_dphi2, dxi2_dq2;
        collision(xw1, xw1_dot, d, n, ddot, tdot, xi2,
                    dd_dxw1, dd_dq2,
                    dn_dxw1, dn_dq2,
                    dddot_dxw1, dddot_dxw1dot,
                    dddot_dq2, dddot_dphi2,
                    dtdot_dxw1, dtdot_dxw1dot,
                    dtdot_dq2, dtdot_dphi2,
                    dxi2_dxw1, dxi2_dq2);

        // test time derivatives
        {
            dtype d_ori, ddot_ori;
            Vector3 n_ori, tdot_ori, xi2_ori;
            collision(xw1, xw1_dot, d_ori, n_ori, ddot_ori, tdot_ori, xi2_ori);

            dtype ddot_fd;

            Vector3 xw1_pos = xw1 + eps * xw1_dot;
            Vector6 dq = eps * phi2;
            Matrix4 E2_pos = E2 * math::exp(dq);
            this->_E_0i = E2_pos;
            dtype d_pos, ddot_pos;
            Vector3 n_pos, tdot_pos, xi2_pos;
            collision(xw1_pos, xw1_dot, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
            ddot_fd = (d_pos - d) / eps;
            this->_E_0i = E2;

            print_error("ddot", ddot_ori, ddot_fd);
        }

        // test dxw1 related
        RowVector3 dd_dxw1_fd, dddot_dxw1_fd;
        Matrix3 dn_dxw1_fd, dtdot_dxw1_fd, dxi2_dxw1_fd;
        for (int i = 0;i < 3;i++) {
            Vector3 xw1_pos = xw1;
            xw1_pos[i] += eps;
            dtype d_pos, ddot_pos;
            Vector3 n_pos, tdot_pos, xi2_pos;
            collision(xw1_pos, xw1_dot, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
            dd_dxw1_fd[i] = (d_pos - d) / eps;
            dddot_dxw1_fd[i] = (ddot_pos - ddot) / eps;
            dn_dxw1_fd.col(i) = (n_pos - n) / eps;
            dtdot_dxw1_fd.col(i) = (tdot_pos - tdot) / eps;
            dxi2_dxw1_fd.col(i) = (xi2_pos - xi2) / eps;
        }
        print_error("dd_dxw1", dd_dxw1, dd_dxw1_fd);
        print_error("dddot_dxw1", dddot_dxw1, dddot_dxw1_fd);
        print_error("dn_dxw1", dn_dxw1, dn_dxw1_fd);
        print_error("dtdot_dxw1", dtdot_dxw1, dtdot_dxw1_fd);
        print_error("dxi2_dxw1", dxi2_dxw1, dxi2_dxw1_fd);

        // test dxw1dot related
        RowVector3 dddot_dxw1dot_fd;
        Matrix3 dtdot_dxw1dot_fd;
        for (int i = 0;i < 3;i++) {
            Vector3 xw1dot_pos = xw1_dot;
            xw1dot_pos[i] += eps;
            dtype d_pos, ddot_pos;
            Vector3 n_pos, tdot_pos, xi2_pos;
            collision(xw1, xw1dot_pos, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
            dddot_dxw1dot_fd[i] = (ddot_pos - ddot) / eps;
            dtdot_dxw1dot_fd.col(i) = (tdot_pos - tdot) / eps;
        }
        print_error("dddot_dxw1dot", dddot_dxw1dot, dddot_dxw1dot_fd);
        print_error("dtdot_dxw1dot", dtdot_dxw1dot, dtdot_dxw1dot_fd);

        // test dq2 related
        RowVector6 dd_dq2_fd, dddot_dq2_fd;
        Matrix36 dn_dq2_fd, dtdot_dq2_fd, dxi2_dq2_fd;
        for (int i = 0;i < 6;i++) {
            Vector6 dq = Vector6::Zero();
            dq[i] = eps;
            Matrix4 E2_pos = E2 * math::exp(dq);
            dtype d_pos, ddot_pos;
            Vector3 n_pos, tdot_pos, xi2_pos;
            this->_E_0i = E2_pos;
            collision(xw1, xw1_dot, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
            dd_dq2_fd[i] = (d_pos - d) / eps;
            dddot_dq2_fd[i] = (ddot_pos - ddot) / eps;
            dn_dq2_fd.col(i) = (n_pos - n) / eps;
            dtdot_dq2_fd.col(i) = (tdot_pos - tdot) / eps;
            dxi2_dq2_fd.col(i) = (xi2_pos - xi2) / eps;
        }
        print_error("dd_dq2", dd_dq2, dd_dq2_fd);
        print_error("dddot_dq2", dddot_dq2, dddot_dq2_fd);
        print_error("dn_dw2", dn_dq2.leftCols(3), dn_dq2_fd.leftCols(3));
        print_error("dn_dv2", dn_dq2.rightCols(3), dn_dq2_fd.rightCols(3));
        print_error("dtdot_dw2", dtdot_dq2.leftCols(3), dtdot_dq2_fd.leftCols(3));
        print_error("dtdot_dv2", dtdot_dq2.rightCols(3), dtdot_dq2_fd.rightCols(3));
        print_error("dxi2_dq2", dxi2_dq2, dxi2_dq2_fd);
        this->_E_0i = E2;

        // test dphi2 related
        RowVector6 dddot_dphi2_fd;
        Matrix36 dtdot_dphi2_fd;
        for (int i = 0;i < 6;i++) {
            Vector6 phi2_pos = phi2;
            phi2_pos[i] += eps;
            this->_phi = phi2_pos;
            dtype d_pos, ddot_pos;
            Vector3 n_pos, tdot_pos, xi2_pos;
            collision(xw1, xw1_dot, d_pos, n_pos, ddot_pos, tdot_pos, xi2_pos);
            dddot_dphi2_fd[i] = (ddot_pos - ddot) / eps;
            dtdot_dphi2_fd.col(i) = (tdot_pos - tdot) / eps;
        }
        print_error("dddot_dw2dot", dddot_dphi2.head(3), dddot_dphi2_fd.head(3));
        print_error("dddot_dv2dot", dddot_dphi2.tail(3), dddot_dphi2_fd.tail(3));
        print_error("dtdot_dw2dot", dtdot_dphi2.leftCols(3), dtdot_dphi2_fd.leftCols(3));
        print_error("dtdot_dv2dot", dtdot_dphi2.rightCols(3), dtdot_dphi2_fd.rightCols(3));
        this->_phi = phi2;
    }
}

}