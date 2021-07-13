// @ Copyright 2016 Massachusetts Institute of Technology.
// 
// This program is free software; you can redistribute it and / or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
// MA 02110-1301, USA.
//
//
// A checkerboard example. Press p to pause/resume.
#include "opengl_viewer.h"

namespace test_opengl_viewer {

class SinusoidalAnimator : public opengl_viewer::Animator {
public:
  SinusoidalAnimator(const Eigen::Vector3f& init_position)
    : init_position_(init_position) {}

  const Eigen::Matrix4f AnimatedModelMatrix(const float t) {
    // Sinusoidal motion.
    return opengl_viewer::Translate(std::sin(0.1f * t)
      * Eigen::Vector3f::UnitZ() + init_position_);
  }
private:
  const Eigen::Vector3f init_position_;
};

class SinusoidalSpanningAnimator : public opengl_viewer::Animator {
public:
  SinusoidalSpanningAnimator(const Eigen::Vector3f& init_position)
    : init_position_(init_position) {}

  const Eigen::Matrix4f AnimatedModelMatrix(const float t) {
    // Sinusoidal motion.
    return opengl_viewer::Translate(std::sin(0.1f * t)
      * Eigen::Vector3f::UnitZ() + init_position_) *
      opengl_viewer::Rotate(10.0f * t, Eigen::Vector3f::UnitY());
  }

private:
  const Eigen::Vector3f init_position_;
};

class CheckerboardKeyboardHandler : public opengl_viewer::KeyboardHandler {
public:
  CheckerboardKeyboardHandler()
    : opengl_viewer::KeyboardHandler(), paused_(false) {}

  void Initialize(const bool paused) { paused_ = paused; }

  void KeyCallback(const int key, const int action) {
    if (key == GLFW_KEY_P && action == GLFW_PRESS) paused_ = !paused_;
  }

  const bool paused() const { return paused_; }

private:
  bool paused_;
};

class CheckerboardTimer : public opengl_viewer::Timer {
public:
  CheckerboardTimer()
    : opengl_viewer::Timer(), fps_(25), dt_(0.04f), current_time_(0.0f),
    keyboard_handler_(NULL) {}

  void Initialize(const int fps,
    const CheckerboardKeyboardHandler* keyboard_handler) {
    fps_ = fps;
    dt_ = 1.0f / fps;
    current_time_ = 0.0f;
    keyboard_handler_ = keyboard_handler;
  }

  const float CurrentTime() {
    if (!keyboard_handler_->paused()) current_time_ += dt_;
    return current_time_;
  }

private:
  int fps_;
  float dt_;
  float current_time_;

  const CheckerboardKeyboardHandler* keyboard_handler_;
};

void Checkerboard() {
  opengl_viewer::Viewer& viewer = opengl_viewer::Viewer::GetViewer();
  opengl_viewer::Option option;
  option.SetFloatOption("camera field of view", 60.0f);
  option.SetFloatOption("camera aspect ratio", 4.0f / 3.0f);
  option.SetIntOption("height", 768);
  option.SetIntOption("width", 1024);
  option.SetFloatOption("shadow acne bias", 0.005f);
  option.SetFloatOption("shadow sampling angle", 0.57f);
  option.SetIntOption("shadow sampling number", 2);

  // Set up timer and keyboard callback.
  CheckerboardKeyboardHandler key_handler;
  key_handler.Initialize(false);
  CheckerboardTimer timer;
  timer.Initialize(25, &key_handler);
  option.SetPointerOption("timer", static_cast<void*>(&timer));
  option.SetPointerOption("keyboard handler",
    static_cast<void*>(&key_handler));
  opengl_viewer::SampleImGuiWrapper imgui_wrapper;
  option.SetPointerOption("imgui wrapper",
    static_cast<void*>(&imgui_wrapper));
  viewer.Initialize(option);

  // A checkerboard.
  const int checker_size = 512, square_size = 32;
  std::vector<std::vector<Eigen::Vector3f>> checker_image(checker_size);
  for (int i = 0; i < checker_size; ++i) {
    checker_image[i] = std::vector<Eigen::Vector3f>(checker_size);
    for (int j = 0; j < checker_size; ++j) {
      // Determine the color of the checker.
      if ((i / square_size - j / square_size) % 2) {
        checker_image[i][j] = Eigen::Vector3f(157.0f, 150.0f, 143.0f) / 255.0f;
      } else {
        checker_image[i][j] = Eigen::Vector3f(216.0f, 208.0f, 197.0f) / 255.0f;
      }
    }
  }
  opengl_viewer::Image checker_texture;
  checker_texture.Initialize(checker_image);

  // A simple ground.
  const Eigen::Matrix3Xf vertex = (Eigen::Matrix<float, 3, 4>()
    << -1.0f, 1.0f, -1.0f, 1.0f,
      0.0f, 0.0f, 0.0f, 0.0f,
      -1.0f, -1.0f, 1.0f, 1.0f).finished();
  const Eigen::Matrix3Xi face = (Eigen::Matrix<int, 3, 2>()
    << 0, 0,
      2, 3,
      3, 1).finished();
  const Eigen::Matrix2Xf uv = (Eigen::Matrix<float, 2, 4>()
    << 0.0f, 0.0f, 1.0f, 1.0f,
      0.0f, 1.0f, 0.0f, 1.0f).finished();
  const Eigen::Matrix4f scale_matrix =
    Eigen::Vector4f(4.0f, 1.0f, 4.0f, 1.0f).asDiagonal();
  opengl_viewer::Option object_option;
  object_option.SetMatrixOption("model matrix", scale_matrix);
  object_option.SetVectorOption("ambient", 0.7f, 0.7f, 0.7f);
  object_option.SetVectorOption("diffuse", 1.0f, 1.0f, 1.0f);
  object_option.SetVectorOption("specular", 1.0f, 1.0f, 1.0f);
  object_option.SetFloatOption("shininess", 1.5f);
  object_option.SetMatrixOption("uv", uv);
  object_option.SetMatrixOption("texture", checker_texture.rgb_data());
  object_option.SetIntOption("texture row num", checker_size);
  object_option.SetIntOption("texture col num", checker_size);
  object_option.SetStringOption("texture mag filter", "nearest");
  viewer.AddStaticObject(vertex, face, object_option);

  // A moving sphere.
  Eigen::Matrix3Xf sphere_vertex;
  Eigen::Matrix3Xi sphere_face;
  Eigen::Matrix2Xf sphere_uv;
  opengl_viewer::Sphere(0.6f, 40, 20, sphere_vertex, sphere_face, sphere_uv);
  opengl_viewer::Rotate(90.0f, Eigen::Vector3f::UnitX(), sphere_vertex);
  const std::string root_folder = std::string(GRAPHICS_CODEBASE_SOURCE_DIR);
  opengl_viewer::Image world_map_texture;
  world_map_texture.Initialize(root_folder +
    "/resources/textures/world_map.jpg");
  object_option.Clear();
  object_option.SetVectorOption("ambient", 0.12f, 0.33f, 0.17f);
  object_option.SetVectorOption("diffuse", 0.44f, 0.56f, 0.54f);
  object_option.SetVectorOption("specular", 0.17f, 0.84f, 0.82f);
  object_option.SetFloatOption("shininess", 4.0f);
  object_option.SetMatrixOption("uv", sphere_uv);
  object_option.SetMatrixOption("texture", world_map_texture.rgb_data());
  object_option.SetIntOption("texture row num", world_map_texture.row_num());
  object_option.SetIntOption("texture col num", world_map_texture.col_num());
  SinusoidalSpanningAnimator sphere_animator(Eigen::Vector3f::UnitY() * 0.6f);
  viewer.AddDynamicObject(sphere_vertex, sphere_face, &sphere_animator,
    object_option);

  // A cylinder.
  Eigen::Matrix3Xf cylinder_vertex;
  Eigen::Matrix3Xi cylinder_face;
  opengl_viewer::Cylinder(0.75f, 0.2f, 2.0f, 64, 8,
    cylinder_vertex, cylinder_face);
  object_option.Clear();
  object_option.SetMatrixOption("model matrix",
    opengl_viewer::Translate(1.2f, 1e-3f, -1.4f) *
    opengl_viewer::Rotate(-90.0f, Eigen::Vector3f::UnitX()));
  object_option.SetVectorOption("ambient", 0.52f, 0.80f, 0.98f);
  object_option.SetVectorOption("diffuse", 0.52f, 0.80f, 0.98f);
  object_option.SetVectorOption("specular", 0.52f, 0.80f, 0.98f);
  object_option.SetFloatOption("shininess", 4.0f);
  viewer.AddStaticObject(cylinder_vertex, cylinder_face, object_option);

  // A cube.
  Eigen::Matrix3Xf cube_vertex;
  Eigen::Matrix3Xi cube_face;
  Eigen::Matrix2Xf cube_uv;
  opengl_viewer::ReadFromObjFile(root_folder + "/resources/meshes/cube.obj",
    cube_vertex, cube_face, cube_uv);
  object_option.SetBoolOption("smooth normal", false);
  object_option.SetMatrixOption("model matrix",
    opengl_viewer::Translate(-1.3f, 0.5f + 1e-3f, 1.5f) *
    opengl_viewer::Rotate(25.0f, Eigen::Vector3f::UnitY()));
  object_option.SetVectorOption("ambient", 0.64f, 0.23f, 0.19f);
  object_option.SetVectorOption("diffuse", 0.94f, 0.33f, 0.19f);
  object_option.SetVectorOption("specular", 0.84f, 0.23f, 0.19f);
  viewer.AddStaticObject(cube_vertex, cube_face, object_option);

  // A point light.
  const Eigen::Vector3f light0_position(0.0f, 1.9f, 0.5f);
  opengl_viewer::Option light_option;
  light_option.SetVectorOption("ambient", 0.17f, 0.15f, 0.14f);
  light_option.SetVectorOption("diffuse", 0.37f, 0.35f, 0.34f);
  light_option.SetVectorOption("specular", 0.37f, 0.35f, 0.34f);
  viewer.AddStaticPointLight(light0_position, light_option);
  // Another point light.
  const Eigen::Vector3f light1_position(-0.5f, 2.1f, -0.5f);
  viewer.AddStaticPointLight(light1_position, light_option);
  // Add a dynamic light.
  SinusoidalAnimator light_animator(Eigen::Vector3f(0.6f, 1.2f, 0.8f));
  viewer.AddDynamicPointLight(&light_animator, light_option);

  viewer.Run();
  viewer.Cleanup();
}

}