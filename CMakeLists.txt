cmake_minimum_required(VERSION 3.5)
project(gen3_controller LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

option(USE_CONAN "Use Conan to fetch Kortex API" ON)

if(USE_CONAN)
  include(${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/conan.cmake)
  conan_check(REQUIRED)
  conan_add_remote(NAME kinova_public
                   URL https://artifactory.kinovaapps.com/artifactory/api/conan/conan-local-public)
  conan_cmake_run(REQUIRES kortex_api_cpp/2.7.0-r.5@kortex/stable
                  PROFILE_AUTO build_type
                  BASIC_SETUP
                  UPDATE)
endif()

add_executable(gen3_controller
  src/Gen3RobotController.cpp)

target_include_directories(gen3_controller PRIVATE include)

if(USE_CONAN)
  target_link_libraries(gen3_controller PRIVATE ${CONAN_LIBS})
endif()

if(UNIX)
  target_link_libraries(gen3_controller PRIVATE pthread)
elseif(WIN32)
  target_link_libraries(gen3_controller PRIVATE winMM ws2_32)
endif()
