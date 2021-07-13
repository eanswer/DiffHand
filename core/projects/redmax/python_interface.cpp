#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "Simulation.h"
#include "SimEnvGenerator.h"

using namespace redmax;

namespace py = pybind11;

Simulation* make_sim(std::string env_name, std::string integrator = "BDF2") {
    Simulation* sim = nullptr;
    if (env_name == "SinglePendulum-Test") {
        sim = SimEnvGenerator::createSinglePendulumTest(integrator);
    } else if (env_name == "Prismatic-Test") {
        sim = SimEnvGenerator::createPrismaticTest(integrator);
    } else if (env_name == "Free2D-Test") {
        sim = SimEnvGenerator::createFree2DTest(integrator);
    } else if (env_name == "GroundContact-Test") {
        sim = SimEnvGenerator::createGroundContactTest(integrator);
    } else if (env_name == "BoxContact-Test") {
        sim = SimEnvGenerator::createBoxContactTest(integrator);
    } else if (env_name == "TorqueFingerFlick-Demo") {
        sim = SimEnvGenerator::createTorqueFingerFlickDemo(integrator);
    } else if (env_name == "TorqueFinger-Demo") {
        sim = SimEnvGenerator::createTorqueFingerDemo(integrator);
    } else {
        return NULL;
    }
    return sim;
}

PYBIND11_MODULE(redmax_py, m) {

    // simulation options
    py::class_<Simulation::Options>(m, "Options")
        .def(py::init<Vector3, dtype, string>(), 
                py::arg("gravity") = -980. * Vector3::UnitZ(), 
                py::arg("h") = 0.02, 
                py::arg("integrator") = "BDF1")
                
        .def_readwrite("gravity", &Simulation::Options::_gravity)
        .def_readwrite("h", &Simulation::Options::_h)
        .def_readwrite("integrator", &Simulation::Options::_integrator);
    
    // viewer options
    py::class_<Simulation::ViewerOptions>(m, "ViewerOptions")
        .def(py::init())

        .def_readwrite("fps", &Simulation::ViewerOptions::_fps)
        .def_readwrite("speed", &Simulation::ViewerOptions::_speed)
        .def_readwrite("camera_pos", &Simulation::ViewerOptions::_camera_pos)
        .def_readwrite("camera_up", &Simulation::ViewerOptions::_camera_up)
        .def_readwrite("camera_lookat", &Simulation::ViewerOptions::_camera_lookat)
        .def_readwrite("ground", &Simulation::ViewerOptions::_ground)
        .def_readwrite("E_g", &Simulation::ViewerOptions::_E_g)
        .def_readwrite("record", &Simulation::ViewerOptions::_record)
        .def_readwrite("record_folder", &Simulation::ViewerOptions::_record_folder)
        .def_readwrite("loop", &Simulation::ViewerOptions::_loop)
        .def_readwrite("infinite", &Simulation::ViewerOptions::_infinite);

    // backward data
    py::class_<BackwardInfo>(m, "BackwardInfo")
        .def_readwrite("df_dq0", &BackwardInfo::_df_dq0)
        .def_readwrite("df_dqdot0", &BackwardInfo::_df_dqdot0)
        .def_readwrite("df_dp", &BackwardInfo::_df_dp)
        .def_readwrite("df_du", &BackwardInfo::_df_du)
        .def_readwrite("df_dq", &BackwardInfo::_df_dq)
        .def_readwrite("df_dvar", &BackwardInfo::_df_dvar)

        .def("set_flags", &BackwardInfo::set_flags,
                "set the flags for backward", 
                py::arg("flag_q0"), py::arg("flag_qdot0"), py::arg("flag_p"), py::arg("flag_u"));

    // backward results
    py::class_<BackwardResults>(m, "BackwardResults")
        .def_readonly("df_dq0", &BackwardResults::_df_dq0)
        .def_readonly("df_dqdot0", &BackwardResults::_df_dqdot0)
        .def_readonly("df_dp", &BackwardResults::_df_dp)
        .def_readonly("df_du", &BackwardResults::_df_du);

    py::class_<Simulation>(m, "Simulation")
        .def(py::init<std::string, bool>(), 
                py::arg("xml_file_path"), 
                py::arg("verbose") = false)

        .def_readonly("options", &Simulation::_options)
        .def_readonly("viewer_options", &Simulation::_viewer_options)

        .def_readonly("ndof_r", &Simulation::_ndof_r)
        .def_readonly("ndof_m", &Simulation::_ndof_m)
        .def_readonly("ndof_p", &Simulation::_ndof_p)
        .def_readonly("ndof_u", &Simulation::_ndof_u)
        .def_readonly("ndof_var", &Simulation::_ndof_var)

        .def("init", &Simulation::init,
                "initialize the simulation")
        .def("get_q_init", &Simulation::get_q_init,
                "get init q for simulation")
        .def("get_qdot_init", &Simulation::get_qdot_init,
                "get init qdot for simulation")
        .def("set_state_init", &Simulation::set_state_init, 
                "set init state (q, qdot) for simulation", 
                py::arg("q_init"), py::arg("qdot_init"))
        .def("set_q_init", &Simulation::set_q_init, 
                "set init q for simulation",
                py::arg("q_init"))
        .def("set_qdot_init", &Simulation::set_qdot_init, 
                "set init qdot for simulation",
                py::arg("qdot_init"))
        .def_property_readonly("q_init", &Simulation::get_q_init)
        .def_property_readonly("qdot_init", &Simulation::get_qdot_init)

        .def("get_q", &Simulation::get_q,
                "get q")
        .def("get_qdot", &Simulation::get_qdot,
                "get qdot")
        .def("get_variables", &Simulation::get_variables,
                "get variables")
        .def("get_design_params", &Simulation::get_design_params,
                "get design parameters")
        .def_property_readonly("q", &Simulation::get_q)
        .def_property_readonly("qdot", &Simulation::get_qdot)
        .def_property_readonly("variables", &Simulation::get_variables)
        .def_property_readonly("design_params", &Simulation::get_design_params)

        .def("set_design_params", &Simulation::set_design_params,
                "set design parameters",
                py::arg("design_params"))

        .def("set_u", &Simulation::set_u,
                "set u",
                py::arg("u"))
        .def("get_ctrl_range", &Simulation::get_ctrl_range, 
                "get the range of the ctrl",
                py::arg("ctrl_min"), py::arg("ctrl_max"))
        .def("print_ctrl_info", &Simulation::print_ctrl_info)
        .def("print_design_params_info", &Simulation::print_design_params_info)

        .def("set_contact_scale", &Simulation::set_contact_scale,
            "set the contact scale for continuation method",
            py::arg("scale"))
            
        .def("set_rendering_mesh_vertices", &Simulation::set_rendering_mesh_vertices,
            "set the rendering mesh vertices for abstract body",
            py::arg("Vs"))
        .def("set_rendering_mesh", &Simulation::set_rendering_mesh,
            "set the rendering mesh for abstract body",
            py::arg("Vs"), py::arg("Fs"))
        
        .def("update_virtual_object", &Simulation::update_virtual_object,
                "update the properties of the virtual objects by name",
                py::arg("name"), py::arg("data"))
                
        .def_readonly("backward_info", &Simulation::_backward_info)
        .def_readonly("backward_results", &Simulation::_backward_results)

        .def("reset", &Simulation::reset,
                "reset the simulation.",
                py::arg("backward_flag") = false, py::arg("backward_design_params_flag") = true)
        .def("forward", &Simulation::forward,
                "step forward the simulation.",
                py::arg("num_steps"), py::arg("verbose") = false, py::arg("test_derivatives") = false)
        .def("backward", &Simulation::backward,
                "backward differentiation.")
        .def("replay", &Simulation::replay,
                "replay (render) the simulation.")

        .def("export_replay", &Simulation::export_replay,
                "export the body transformations in each step.",
                py::arg("folder"))

        .def("print_time_report", &Simulation::print_time_report,
                "print time report of the simulation.");

    m.def("make_sim", &make_sim, "initialize a simulation instance", py::arg("env_name"), py::arg("integrator") = "BDF2");
}