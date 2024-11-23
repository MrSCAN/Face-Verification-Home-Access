cmake_minimum_required(VERSION 3.12)

# Project name and language
project(FaceRecognitionApp LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the source directory and the destination directory
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/models)
set(DEST_DIR ${CMAKE_BINARY_DIR}/models)

# Copy the directory to the build directory
file(COPY ${SOURCE_DIR} DESTINATION ${DEST_DIR})

# Define the target executable name
set(TARGET FaceRecognitionApp)
set(API_TARGET api)
add_subdirectory(Crow)
include_directories(
    ${PROJECT_SOURCE_DIR}
    /usr/include # Add ASIO include path (adjust this path based on your installation)
)
# Define the source files
set(API_SRC_FILES
    api.cc
    face_features.cc
    face_features.hh
    db_utils.cc
    db_utils.hh
)

set(SRC_FILES_MAIN
    main_pi.cc
    face_features.cc
    face_features.hh
    db_utils.cc
    db_utils.hh
)

# Define output directories for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}
)

# Find required packages (for OpenCV, etc.)
find_package(OpenCV REQUIRED)

# Link libraries
set(LINK_LIBRARIES
    dlib
    pthread
    jpeg
    png
    X11
    openblas
    sqlite3
    Crow
    ${OpenCV_LIBS}
)

# Add executable
add_executable(${API_TARGET} ${API_SRC_FILES})
add_executable(${TARGET} ${SRC_FILES_MAIN})


# Link libraries to the executable
target_link_libraries(${TARGET} PRIVATE ${LINK_LIBRARIES})
target_link_libraries(${API_TARGET} PRIVATE ${LINK_LIBRARIES})
