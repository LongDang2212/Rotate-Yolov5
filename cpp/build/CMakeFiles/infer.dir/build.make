# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/longdd/LongDD/Rotate-Yolov5/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/longdd/LongDD/Rotate-Yolov5/cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/infer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/infer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/infer.dir/flags.make

CMakeFiles/infer.dir/infer_angle.cpp.o: CMakeFiles/infer.dir/flags.make
CMakeFiles/infer.dir/infer_angle.cpp.o: ../infer_angle.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/longdd/LongDD/Rotate-Yolov5/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/infer.dir/infer_angle.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/infer.dir/infer_angle.cpp.o -c /home/longdd/LongDD/Rotate-Yolov5/cpp/infer_angle.cpp

CMakeFiles/infer.dir/infer_angle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/infer.dir/infer_angle.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/longdd/LongDD/Rotate-Yolov5/cpp/infer_angle.cpp > CMakeFiles/infer.dir/infer_angle.cpp.i

CMakeFiles/infer.dir/infer_angle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/infer.dir/infer_angle.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/longdd/LongDD/Rotate-Yolov5/cpp/infer_angle.cpp -o CMakeFiles/infer.dir/infer_angle.cpp.s

CMakeFiles/infer.dir/infer_angle.cpp.o.requires:

.PHONY : CMakeFiles/infer.dir/infer_angle.cpp.o.requires

CMakeFiles/infer.dir/infer_angle.cpp.o.provides: CMakeFiles/infer.dir/infer_angle.cpp.o.requires
	$(MAKE) -f CMakeFiles/infer.dir/build.make CMakeFiles/infer.dir/infer_angle.cpp.o.provides.build
.PHONY : CMakeFiles/infer.dir/infer_angle.cpp.o.provides

CMakeFiles/infer.dir/infer_angle.cpp.o.provides.build: CMakeFiles/infer.dir/infer_angle.cpp.o


# Object files for target infer
infer_OBJECTS = \
"CMakeFiles/infer.dir/infer_angle.cpp.o"

# External object files for target infer
infer_EXTERNAL_OBJECTS =

infer: CMakeFiles/infer.dir/infer_angle.cpp.o
infer: CMakeFiles/infer.dir/build.make
infer: libryolo_trt.so
infer: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.1.1
infer: /usr/local/cuda/lib64/libcudart_static.a
infer: /usr/lib/aarch64-linux-gnu/librt.so
infer: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.1.1
infer: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.1.1
infer: /usr/local/cuda/lib64/libcudart_static.a
infer: /usr/lib/aarch64-linux-gnu/librt.so
infer: CMakeFiles/infer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/longdd/LongDD/Rotate-Yolov5/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable infer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/infer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/infer.dir/build: infer

.PHONY : CMakeFiles/infer.dir/build

CMakeFiles/infer.dir/requires: CMakeFiles/infer.dir/infer_angle.cpp.o.requires

.PHONY : CMakeFiles/infer.dir/requires

CMakeFiles/infer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/infer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/infer.dir/clean

CMakeFiles/infer.dir/depend:
	cd /home/longdd/LongDD/Rotate-Yolov5/cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/longdd/LongDD/Rotate-Yolov5/cpp /home/longdd/LongDD/Rotate-Yolov5/cpp /home/longdd/LongDD/Rotate-Yolov5/cpp/build /home/longdd/LongDD/Rotate-Yolov5/cpp/build /home/longdd/LongDD/Rotate-Yolov5/cpp/build/CMakeFiles/infer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/infer.dir/depend

