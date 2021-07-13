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
#include <iostream>
#include <string>

namespace test_opengl_viewer {

void Checkerboard();
void PointLightShadow();

}

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::cout << "Please provide the name of the scene "
      "(use one of them below):" << std::endl
      << "checkerboard" << std::endl
      << "point_light_shadow" << std::endl;
    return 0;
  }

  const std::string scene_name(argv[1]);
  if (scene_name == "checkerboard") {
    test_opengl_viewer::Checkerboard();
  } else if (scene_name == "point_light_shadow") {
    test_opengl_viewer::PointLightShadow();
  } else {
    std::cout << "Error: unsupported scene." << std::endl;
  }
  return 0;
}
