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
#include "opengl_viewer.h"

namespace test_opengl_viewer {

void PointLightShadow() {
  opengl_viewer::Viewer& viewer = opengl_viewer::Viewer::GetViewer();
  opengl_viewer::Option option;
  option.SetFloatOption("camera field of view", 45.0f);
  option.SetFloatOption("camera aspect ratio", 1.0f);
  option.SetIntOption("height", 800);
  option.SetIntOption("width", 800);
  const Eigen::Vector3f unit_x = Eigen::Vector3f::UnitX(),
    unit_y = Eigen::Vector3f::UnitY(),
    unit_z = Eigen::Vector3f::UnitZ();
  option.SetVectorOption("camera pos", unit_z);
  option.SetVectorOption("camera look at", 0.0f, 0.0f, 0.0f);
  option.SetVectorOption("camera up", unit_y);
  option.SetFloatOption("shadow acne bias", 0.01f);
  option.SetFloatOption("shadow sampling angle", 0.5f);
  option.SetBoolOption("shadow", false);
  viewer.Initialize(option);

  // Add geometry:
  // Ground.
  const Eigen::Vector3f unit_r = unit_x, unit_g = unit_y, unit_b = unit_z;
  const Eigen::Vector3f gray = (unit_r + unit_g + unit_b) * 0.85f;
  const Eigen::Matrix3Xf vertex = (Eigen::Matrix<float, 3, 4>()
    << -1.0f, 1.0f, -1.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f,
    -1.0f, -1.0f, 1.0f, 1.0f).finished();
  const Eigen::Matrix3Xi face = (Eigen::Matrix<int, 3, 2>()
    << 0, 0,
    2, 3,
    3, 1).finished();
  opengl_viewer::Option object_options;
  object_options.SetMatrixOption("model matrix",
    opengl_viewer::Translate(-unit_y));
  object_options.SetVectorOption("ambient", unit_r * 0.5f);
  object_options.SetVectorOption("diffuse", unit_r * 0.5f);
  viewer.AddStaticObject(vertex, face, object_options);

  // Ceiling.
  object_options.SetMatrixOption("model matrix",
    opengl_viewer::Translate(unit_y) *
    opengl_viewer::Rotate(180.0f, unit_x));
  object_options.SetVectorOption("ambient", unit_g * 0.5f);
  object_options.SetVectorOption("diffuse", unit_g * 0.5f);
  viewer.AddStaticObject(vertex, face, object_options);

  // Front.
  object_options.SetMatrixOption("model matrix",
    opengl_viewer::Translate(-unit_z) *
    opengl_viewer::Rotate(90.0f, unit_x));
  object_options.SetVectorOption("ambient", unit_b * 0.5f);
  object_options.SetVectorOption("diffuse", unit_b * 0.5f);
  viewer.AddStaticObject(vertex, face, object_options);

  // Left.
  object_options.SetMatrixOption("model matrix",
    opengl_viewer::Translate(-unit_x) *
    opengl_viewer::Rotate(-90.0f, unit_z));
  object_options.SetVectorOption("ambient", (unit_r + unit_g) * 0.5f);
  object_options.SetVectorOption("diffuse", (unit_r + unit_g) * 0.5f);
  viewer.AddStaticObject(vertex, face, object_options);

  // Right.
  object_options.SetMatrixOption("model matrix",
    opengl_viewer::Translate(unit_x) *
    opengl_viewer::Rotate(90.0f, unit_z));
  object_options.SetVectorOption("ambient", (unit_b + unit_g) * 0.5f);
  object_options.SetVectorOption("diffuse", (unit_b + unit_g) * 0.5f);
  viewer.AddStaticObject(vertex, face, object_options);

  // Ground shadow.
  object_options.SetMatrixOption("model matrix",
    opengl_viewer::Translate(-0.5f * unit_y) *
    opengl_viewer::Scale(0.25f));
  object_options.SetVectorOption("ambient", gray);
  viewer.AddStaticObject(vertex, face, object_options);

  // Ceiling shadow.
  object_options.SetMatrixOption("model matrix",
    opengl_viewer::Translate(0.5f * unit_y) *
    opengl_viewer::Rotate(180.0f, unit_x) *
    opengl_viewer::Scale(0.25f));
  viewer.AddStaticObject(vertex, face, object_options);

  // Front shadow.
  object_options.SetMatrixOption("model matrix",
    opengl_viewer::Translate(-0.5f * unit_z) *
    opengl_viewer::Rotate(90.0f, unit_x) *
    opengl_viewer::Scale(0.25f));
  viewer.AddStaticObject(vertex, face, object_options);

  // Left shadow.
  object_options.SetMatrixOption("model matrix",
    opengl_viewer::Translate(-0.5f * unit_x) *
    opengl_viewer::Rotate(-90.0f, unit_z) *
    opengl_viewer::Scale(0.25f));
  viewer.AddStaticObject(vertex, face, object_options);

  // Right shadow.
  object_options.SetMatrixOption("model matrix",
    opengl_viewer::Translate(0.5f * unit_x) *
    opengl_viewer::Rotate(90.0f, unit_z) *
    opengl_viewer::Scale(0.25f));
  viewer.AddStaticObject(vertex, face, object_options);

  // Add light.
  opengl_viewer::Option light_option;
  light_option.SetVectorOption("ambient", gray * 0.5f);
  light_option.SetVectorOption("diffuse", gray * 0.5f);
  light_option.SetVectorOption("specular", gray * 0.5f);
  viewer.AddStaticPointLight(Eigen::Vector3f(0.25f, 0.0f, 0.0f),
    light_option);
  viewer.AddStaticPointLight(Eigen::Vector3f(0.0f, 0.25f, 0.0f),
    light_option);

  viewer.Run();
  viewer.Cleanup();
}

}
