#include "Body/Body.h"
#include "Body/BodyPrimitiveShape.h"
#include "Robot.h"
#include "Sensor/TactileSensor.h"
#include "Simulation.h"

namespace redmax {

const Eigen::Matrix4f TactileSensorAnimator::AnimatedModelMatrix(const float t) {
    Eigen::Matrix4f model_matrix = _tactile_sensor->_body->_E_0i.cast<float>();
    if (_tactile_sensor->_body->_sim->_options->_unit == "cm-g")
        model_matrix.topRightCorner(3, 1) /= 10.; // scale for better visualization
    else
        model_matrix.topRightCorner(3, 1) *= 10.; // scale for better visualization
    return model_matrix;
}

void TactileSensor::init() {
    _depth.clear(); _normal_force.clear(); _shear_force.clear(); _tactile_force.clear(); _contact_body.clear();
    for (int i = 0; i < _pos_i.size();++i) {
        _depth.push_back(0.);
        _normal_force.push_back(0.);
        _shear_force.push_back(Vector2::Zero());   
        _tactile_force.push_back(Vector3::Zero());
        _contact_body.push_back(nullptr);
    }
}

void TactileSensor::compute_tactile_values() {
    Matrix3 R1 = _body->_E_0i.topLeftCorner(3, 3);
    Matrix3 R1T = R1.transpose();
    Vector3 p1 = _body->_E_0i.topRightCorner(3, 1);
    Vector6 phi1 = _body->_phi;
    for (int i = 0;i < _pos_i.size();++i) {
        _contact_body[i] = nullptr;
        _normal_force[i] = 0.;
        _shear_force[i].setZero();
        _tactile_force[i].setZero();

        Vector3 xi1 = _pos_i[i];
        Vector3 xw1 = R1 * xi1 + p1;
        Vector3 xw1_dot = R1 * (math::skew(xi1).transpose() * phi1.head(3) + phi1.tail(3));
        
        for (auto body : _robot->_bodies) {
            if (dynamic_cast<BodyPrimitiveShape*>(const_cast<Body*>(body)) != nullptr && body != _body) {
                BodyPrimitiveShape* primitive_body = dynamic_cast<BodyPrimitiveShape*>(const_cast<Body*>(body));
                if (primitive_body->distance(xw1) < 0.) {
                    dtype d, ddot;
                    Vector3 n, tdot, xi2;
                    primitive_body->collision(xw1, xw1_dot, d, n, ddot, tdot, xi2);

                    // contact force
                    Vector3 fc = (-_kn * d + _damping * ddot * d) * R1T * n;

                    // frictional force
                    dtype fc_norm = fc.norm();
                    dtype tdot_norm = tdot.norm();

                    // std::cerr << "pos = " << xi1.transpose() << ", fc_norm = " << fc_norm << ", tdot_norm = " << tdot_norm << ", d = " << d << ", ddot = " << ddot << ", xw1 = " << xw1.transpose() << std::endl; 
                    // std::cerr << "xi2 = " << xi2.transpose() << std::endl;
                    Vector3 ft = Vector3::Zero();
                    if (_mu > constants::eps) {
                        if (_mu * fc_norm >= _kt * tdot_norm - constants::eps) {
                            // std::cerr << "tactile contact static " << i << std::endl;
                            // std::cerr << "static " << xi1.transpose() << ", " << fc_norm << ", " << tdot_norm << std::endl;
                            ft = -_kt * R1T * tdot;
                        } else {
                            // std::cerr << "tactile contact dynamic " << i << std::endl;
                            // std::cerr << "dynamic " << xi1.transpose() << ", " << fc_norm << ", " << tdot_norm << std::endl;
                            ft = -_mu * fc_norm / tdot_norm * R1T * tdot;
                        }
                    }

                    _depth[i] = -d;
                    Vector3 force = fc + ft; // contact force in local body frame
                    _shear_force[i](0) = force.dot(_axis0_i[i]);
                    _shear_force[i](1) = force.dot(_axis1_i[i]);
                    _normal_force[i] = -force.dot(_normal_i[i]);
                    _tactile_force[i](0) = _shear_force[i](0);
                    _tactile_force[i](1) = _shear_force[i](1);
                    _tactile_force[i](2) = _normal_force[i];
                    _contact_body[i] = body;
                }
            }
        }
    }
}

void TactileSensor::compute_tactile_values_with_derivatives(MatrixX& dtactile_dqm, MatrixX& dtactile_dphi) {
    dtactile_dqm = MatrixX::Zero(_pos_i.size() * 3, 12);
    dtactile_dphi = MatrixX::Zero(_pos_i.size() * 3, 12);

    Matrix3 R1 = _body->_E_0i.topLeftCorner(3, 3);
    Matrix3 R1T = R1.transpose();
    Vector3 p1 = _body->_E_0i.topRightCorner(3, 1);
    Vector6 phi1 = _body->_phi;
    for (int i = 0;i < _pos_i.size();++i) {
        _contact_body[i] = nullptr;
        _normal_force[i] = 0.;
        _shear_force[i].setZero();
        _tactile_force[i].setZero();

        Vector3 xi1 = _pos_i[i];
        Vector3 xw1 = R1 * xi1 + p1;
        Vector3 xw1_dot = R1 * (math::skew(xi1).transpose() * phi1.head(3) + phi1.tail(3));
        
        Matrix36 dxw1_dq1;
        dxw1_dq1.leftCols(3) = -R1 * math::skew(xi1);
        dxw1_dq1.rightCols(3) = R1;
        
        Matrix36 dxw1dot_dq1;
        dxw1dot_dq1.leftCols(3) = -R1 * math::skew(math::skew(xi1).transpose() * phi1.head(3) + phi1.tail(3));
        dxw1dot_dq1.rightCols(3).setZero();

        Matrix36 dxw1dot_dphi1;
        dxw1dot_dphi1.leftCols(3) = R1 * math::skew(xi1).transpose();
        dxw1dot_dphi1.rightCols(3) = R1;

        Matrix3 proj;
        proj.col(0) = _axis0_i[i];
        proj.col(1) = _axis1_i[i];
        proj.col(2) = -_normal_i[i];

        for (auto body : _robot->_bodies) {
            if (dynamic_cast<BodyPrimitiveShape*>(const_cast<Body*>(body)) != nullptr && body != _body) {
                BodyPrimitiveShape* primitive_body = dynamic_cast<BodyPrimitiveShape*>(const_cast<Body*>(body));
                if (primitive_body->distance(xw1) < 0.) {
                    dtype d, ddot;
                    Vector3 n, tdot, xi2;
                    RowVector3 dd_dxw1, dddot_dxw1, dddot_dxw1dot;
                    RowVector6 dd_dq2, dddot_dq2, dddot_dphi2;
                    Matrix3 dn_dxw1, dtdot_dxw1, dtdot_dxw1dot, dxi2_dxw1;
                    Matrix36 dn_dq2, dtdot_dq2, dtdot_dphi2, dxi2_dq2;
                    primitive_body->collision(xw1, xw1_dot, d, n, ddot, tdot, xi2,
                                                dd_dxw1, dd_dq2,
                                                dn_dxw1, dn_dq2,
                                                dddot_dxw1, dddot_dxw1dot,
                                                dddot_dq2, dddot_dphi2,
                                                dtdot_dxw1, dtdot_dxw1dot,
                                                dtdot_dq2, dtdot_dphi2,
                                                dxi2_dxw1, dxi2_dq2);

                    Vector3 n1 = R1T * n;

                    // contact force
                    Vector3 fc = (-_kn * d + _damping * ddot * d) * n1;

                    // derivatives
                    RowVector6 dd_dq1 = dd_dxw1 * dxw1_dq1;
                    RowVector6 dddot_dq1 = dddot_dxw1 * dxw1_dq1 + dddot_dxw1dot * dxw1dot_dq1;
                    Matrix36 dn_dq1 = dn_dxw1 * dxw1_dq1;
                    RowVector6 dddot_dphi1 = dddot_dxw1dot * dxw1dot_dphi1;
                    Matrix36 dtdot_dq1 = dtdot_dxw1 * dxw1_dq1 + dtdot_dxw1dot * dxw1dot_dq1;
                    Matrix36 dtdot_dphi1 = dtdot_dxw1dot * dxw1dot_dphi1;
                    
                    Matrix36 dfc_dq1 = - n1 * (_kn * dd_dq1 - _damping * dddot_dq1 * d - _damping * ddot * dd_dq1) -
                            (_kn * d - _damping * ddot * d) * R1T * dn_dq1;
                    dfc_dq1.leftCols(3) -= (_kn * d - _damping * ddot * d) * math::skew(R1.transpose() * n);
                    
                    Matrix36 dfc_dq2 = - n1 * (_kn * dd_dq2 - _damping * dddot_dq2 * d - _damping * ddot * dd_dq2) - 
                            (_kn * d - _damping * ddot * d) * R1T * dn_dq2;
                    
                    Matrix36 dfc_dphi1 = n1 * _damping * dddot_dphi1 * d;

                    Matrix36 dfc_dphi2 = n1 * _damping * dddot_dphi2 * d;

                    // frictional force
                    dtype fc_norm = fc.norm();
                    dtype tdot_norm = tdot.norm();
                    Vector3 ft = Vector3::Zero();
                    Matrix36 dft_dq1 = Matrix36::Zero();
                    Matrix36 dft_dq2 = Matrix36::Zero();
                    Matrix36 dft_dphi1 = Matrix36::Zero();
                    Matrix36 dft_dphi2 = Matrix36::Zero();
                    if (_mu > constants::eps) {
                        dtype fc_norm = fc.norm();
                        dtype tdot_norm = tdot.norm();
                        RowVector6 dfc_norm_dq1 = (1. / fc_norm) * (fc.transpose() * dfc_dq1);
                        RowVector6 dfc_norm_dq2 = (1. / fc_norm) * (fc.transpose() * dfc_dq2);
                        RowVector6 dfc_norm_dphi1 = (1. / fc_norm) * (fc.transpose() * dfc_dphi1);
                        RowVector6 dfc_norm_dphi2 = (1. / fc_norm) * (fc.transpose() * dfc_dphi2);
                        RowVector6 dtdot_norm_dq1 = (1. / tdot_norm) * (tdot.transpose() * dtdot_dq1);
                        RowVector6 dtdot_norm_dq2 = (1. / tdot_norm) * (tdot.transpose() * dtdot_dq2);
                        RowVector6 dtdot_norm_dphi1 = (1. / tdot_norm) * (tdot.transpose() * dtdot_dphi1);
                        RowVector6 dtdot_norm_dphi2 = (1. / tdot_norm) * (tdot.transpose() * dtdot_dphi2);

                        if (_mu * fc_norm >= _kt * tdot_norm - constants::eps) {
                            ft = -_kt * R1T * tdot;
                            
                            // derivatives
                            dft_dq1 = -_kt * R1T * dtdot_dq1;
                            dft_dq1.leftCols(3) -= _kt * math::skew(R1T * tdot);
                            dft_dq2 = -_kt * R1T * dtdot_dq2;
                            
                            dft_dphi1 = -_kt * R1T * dtdot_dphi1;
                            dft_dphi2 = -_kt * R1T * dtdot_dphi2;
                        } else {
                            ft = -_mu * fc_norm / tdot_norm * R1T * tdot;

                            // derivatives
                            dft_dq1 = -_mu / tdot_norm * R1T * (tdot * dfc_norm_dq1 - fc_norm / tdot_norm * tdot * dtdot_norm_dq1 + fc_norm * dtdot_dq1);
                            dft_dq1.leftCols(3) -= _mu * math::skew(fc_norm / tdot_norm * R1.transpose() * tdot);
                            dft_dq2 = -_mu / tdot_norm * R1T * (tdot * dfc_norm_dq2 - fc_norm / tdot_norm * tdot * dtdot_norm_dq2 + fc_norm * dtdot_dq2);

                            // std::cerr << "dtdot_dphi1 = " << std::endl << dtdot_dphi1 << std::endl;
                            // std::cerr << "dtdot_norm_dphi1 = " << std::endl << dtdot_norm_dphi1 << std::endl;
                            dft_dphi1 = -_mu / tdot_norm * R1T * (tdot * dfc_norm_dphi1 - fc_norm / tdot_norm * tdot * dtdot_norm_dphi1 + fc_norm * dtdot_dphi1);
                            dft_dphi2 = -_mu / tdot_norm * R1T * (tdot * dfc_norm_dphi2 - fc_norm / tdot_norm * tdot * dtdot_norm_dphi2 + fc_norm * dtdot_dphi2);

                            // std::cerr << "dft_dphi1 = " << std::endl << dft_dphi1 << std::endl;
                        }
                    }

                    Vector3 force = fc + ft; // contact force in local body frame
                    _shear_force[i](0) = force.dot(_axis0_i[i]);
                    _shear_force[i](1) = force.dot(_axis1_i[i]);
                    _normal_force[i] = -force.dot(_normal_i[i]);
                    _tactile_force[i](0) = _shear_force[i](0);
                    _tactile_force[i](1) = _shear_force[i](1);
                    _tactile_force[i](2) = _normal_force[i];
                    _contact_body[i] = body;

                    Matrix36 df_dq1 = dft_dq1 + dfc_dq1;
                    Matrix36 df_dq2 = dft_dq2 + dfc_dq2;
                    Matrix36 df_dphi1 = dft_dphi1 + dfc_dphi1;
                    Matrix36 df_dphi2 = dft_dphi2 + dfc_dphi2;
                    dtactile_dqm.block(i * 3, 0, 3, 6) = proj.transpose() * df_dq1;
                    dtactile_dqm.block(i * 3, 6, 3, 6) = proj.transpose() * df_dq2;
                    dtactile_dphi.block(i * 3, 0, 3, 6) = proj.transpose() * df_dphi1;
                    dtactile_dphi.block(i * 3, 6, 3, 6) = proj.transpose() * df_dphi2;
                }
            }
        }
    }
}

void TactileSensor::test_derivatives_runtime() {
    // std::cerr << "**************************** Tactile Derivatives ***************************" << std::endl;
// 
    dtype eps = 1e-8;

    for (int ii = 0; ii < 1;++ii) {
        // printf("---------------------- eps = %.9lf ----------------------------\n", eps);

        MatrixX dtactile_dqm, dtactile_dphi;

        compute_tactile_values_with_derivatives(dtactile_dqm, dtactile_dphi);
        VectorX tactile_force(_tactile_force.size() * 3);
        for (int i = 0;i < _tactile_force.size();++i)
            tactile_force.segment(i * 3, 3) = _tactile_force[i];
        MatrixX dtactile_dqm_full = MatrixX::Zero(tactile_force.size(), _robot->_ndof_m);
        MatrixX dtactile_dphi_full = MatrixX::Zero(tactile_force.size(), _robot->_ndof_m);
        for (int i = 0;i < _tactile_force.size();++i) 
            if (_contact_body[i] != nullptr) {
                dtactile_dqm_full.block(i * 3, _body->_index[0], 3, 6) = dtactile_dqm.block(i * 3, 0, 3, 6);
                dtactile_dqm_full.block(i * 3, _contact_body[i]->_index[0], 3, 6) = dtactile_dqm.block(i * 3, 6, 3, 6);
                dtactile_dphi_full.block(i * 3, _body->_index[0], 3, 6) = dtactile_dphi.block(i * 3, 0, 3, 6);
                dtactile_dphi_full.block(i * 3, _contact_body[i]->_index[0], 3, 6) = dtactile_dphi.block(i * 3, 6, 3, 6);
            }

        MatrixX dtactile_dqm_full_fd = MatrixX::Zero(tactile_force.size(), _robot->_ndof_m);
        MatrixX dtactile_dphi_full_fd = MatrixX::Zero(tactile_force.size(), _robot->_ndof_m);
        for (auto body : _robot->_bodies) {
            Matrix4 E = body->_E_0i;
            for (int i = 0;i < 6;++i) {
                Vector6 dq = Vector6::Zero();
                dq[i] = eps;
                Matrix4 E_pos = E * math::exp(dq);
                body->_E_0i = E_pos;
                compute_tactile_values();
                VectorX tactile_force_pos(_tactile_force.size() * 3);
                for (int j = 0;j < _tactile_force.size();++j) {
                    tactile_force_pos.segment(j * 3, 3) = _tactile_force[j];
                }
                dtactile_dqm_full_fd.col(body->_index[i]) = (tactile_force_pos - tactile_force) / eps;
            }
            body->_E_0i = E;

            Vector6 phi = body->_phi;
            for (int i = 0;i < 6;++i) {
                body->_phi = phi;
                body->_phi[i] += eps;
                compute_tactile_values();
                VectorX tactile_force_pos(_tactile_force.size() * 3);
                for (int j = 0;j < _tactile_force.size();++j) {
                    tactile_force_pos.segment(j * 3, 3) = _tactile_force[j];
                }
                dtactile_dphi_full_fd.col(body->_index[i]) = (tactile_force_pos - tactile_force) / eps;
            }
            body->_phi = phi;
        }
        // for (int i = 0;i < dtactile_dqm_full.rows();++i) {
        //     if ((dtactile_dqm_full.row(i) - dtactile_dqm_full_fd.row(i)).norm() > 1.) {
        //         std::cerr << "row " << i << std::endl;
        //         std::cerr << "analytical = " << std::endl;
        //         std::cerr << dtactile_dqm_full.row(i) << std::endl;
        //         std::cerr << "finite difference = " << std::endl;
        //         std::cerr << dtactile_dqm_full_fd.row(i) << std::endl;
        //         break;
        //     }
        // }
        // print_error_full("dtactile_dqm", dtactile_dqm_full, dtactile_dqm_full_fd);
        // // std::cerr << "ana" << std::endl << dtactile_dphi_full << std::endl;
        // // std::cerr << "fd" << std::endl << dtactile_dphi_full_fd << std::endl;
        // print_error_full("dtactile_dphi", dtactile_dphi_full, dtactile_dphi_full_fd);

        print_error("dtactile_dqm", dtactile_dqm_full, dtactile_dqm_full_fd);
        // std::cerr << "ana" << std::endl << dtactile_dphi_full << std::endl;
        // std::cerr << "fd" << std::endl << dtactile_dphi_full_fd << std::endl;
        print_error("dtactile_dphi", dtactile_dphi_full, dtactile_dphi_full_fd);

        eps /= 10.;
    }
}

// rendering
void TactileSensor::get_rendering_objects(
    std::vector<Matrix3Xf>& vertex_list, 
    std::vector<Matrix3Xi>& face_list,
    std::vector<opengl_viewer::Option>& option_list,
    std::vector<opengl_viewer::Animator*>& animator_list) {
    
    if (_render) {
        Matrix3Xf vertex;
        Matrix3Xi face;
        Matrix2Xf uv;
        opengl_viewer::Option object_option;

        opengl_viewer::ReadFromObjFile(
            std::string(GRAPHICS_CODEBASE_SOURCE_DIR) + "/resources/meshes/low_res_sphere.obj",
            vertex, face, uv);

        vertex *= (float)0.0004;

        Matrix3Xf all_vertices = Matrix3Xf::Zero(3, vertex.cols() * _pos_i.size());
        Matrix3Xi all_faces = Matrix3Xi::Zero(3, face.cols() * _pos_i.size());

        for (int i = 0;i < _pos_i.size();i++) {
            all_vertices.middleCols(i * vertex.cols(), vertex.cols()) = vertex.colwise() + _pos_i[i].cast<float>();
            all_faces.middleCols(i * face.cols(), face.cols()) = face + Matrix3Xi::Constant(3, face.cols(), vertex.cols() * i);
        }

        _rendering_vertices = all_vertices;
        _rendering_faces = all_faces;
        
        if (_body->_sim->_options->_unit == "cm-g")
            all_vertices /= 10.;
        else
            all_vertices *= 10.;

        object_option.SetBoolOption("smooth normal", false);
        object_option.SetVectorOption("ambient", 0.8, 0.8, 0.8);
        object_option.SetVectorOption("diffuse", 0.4f, 0.2368f, 0.1036f);
        object_option.SetVectorOption("specular", 0.774597f, 0.458561f, 0.200621f);
        object_option.SetFloatOption("shininess", 76.8f);

        _animator = new TactileSensorAnimator(this);

        vertex_list.push_back(all_vertices);
        face_list.push_back(all_faces);
        option_list.push_back(object_option);
        animator_list.push_back(_animator);
    }
}

std::vector<Vector3> TactileSensor::get_tactile_sensor_pos() {
    return _pos_i;
}

void TactileSensor::update_tactile_sensor_pos(std::vector<Vector3> new_pos_i) {
    _pos_i = new_pos_i;
}

};