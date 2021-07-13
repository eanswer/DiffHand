set(TEST_OPENGL_VIEWER_DIR ${CMAKE_CURRENT_LIST_DIR})

file(GLOB_RECURSE TEST_OPENGL_VIEWER_HEADER ${TEST_OPENGL_VIEWER_DIR}/*.h)
file(GLOB_RECURSE TEST_OPENGL_VIEWER_SOURCE ${TEST_OPENGL_VIEWER_DIR}/*.cpp)

add_executable(test_opengl_viewer ${TEST_OPENGL_VIEWER_HEADER} ${TEST_OPENGL_VIEWER_SOURCE})

# Add include directories.
target_include_directories(test_opengl_viewer PRIVATE ${TEST_OPENGL_VIEWER_DIR}/include)

# Add libs.
target_link_libraries(test_opengl_viewer
  opengl_viewer
)
