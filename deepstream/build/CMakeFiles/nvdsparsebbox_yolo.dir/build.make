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
CMAKE_SOURCE_DIR = /home/longdd/Documents/ryolov5_jetson/deepstream

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/longdd/Documents/ryolov5_jetson/deepstream/build

# Include any dependencies generated for this target.
include CMakeFiles/nvdsparsebbox_yolo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/nvdsparsebbox_yolo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nvdsparsebbox_yolo.dir/flags.make

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o: CMakeFiles/nvdsparsebbox_yolo.dir/flags.make
CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o: /home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/longdd/Documents/ryolov5_jetson/deepstream/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu -o CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o.requires:

.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o.requires

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o.provides: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o.requires
	$(MAKE) -f CMakeFiles/nvdsparsebbox_yolo.dir/build.make CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o.provides.build
.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o.provides

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o.provides.build: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o


CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o: CMakeFiles/nvdsparsebbox_yolo.dir/flags.make
CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o: /home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/longdd/Documents/ryolov5_jetson/deepstream/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu -o CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o.requires:

.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o.requires

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o.provides: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o.requires
	$(MAKE) -f CMakeFiles/nvdsparsebbox_yolo.dir/build.make CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o.provides.build
.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o.provides

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o.provides.build: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o


CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o: CMakeFiles/nvdsparsebbox_yolo.dir/flags.make
CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o: /home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/longdd/Documents/ryolov5_jetson/deepstream/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o -c /home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp > CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.i

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp -o CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.s

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o.requires:

.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o.requires

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o.provides: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o.requires
	$(MAKE) -f CMakeFiles/nvdsparsebbox_yolo.dir/build.make CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o.provides.build
.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o.provides

CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o.provides.build: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o


CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o: CMakeFiles/nvdsparsebbox_yolo.dir/flags.make
CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o: ../nvdsparsebbox_yolo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/longdd/Documents/ryolov5_jetson/deepstream/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o -c /home/longdd/Documents/ryolov5_jetson/deepstream/nvdsparsebbox_yolo.cpp

CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/longdd/Documents/ryolov5_jetson/deepstream/nvdsparsebbox_yolo.cpp > CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.i

CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/longdd/Documents/ryolov5_jetson/deepstream/nvdsparsebbox_yolo.cpp -o CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.s

CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o.requires:

.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o.requires

CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o.provides: CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o.requires
	$(MAKE) -f CMakeFiles/nvdsparsebbox_yolo.dir/build.make CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o.provides.build
.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o.provides

CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o.provides.build: CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o


# Object files for target nvdsparsebbox_yolo
nvdsparsebbox_yolo_OBJECTS = \
"CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o" \
"CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o" \
"CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o" \
"CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o"

# External object files for target nvdsparsebbox_yolo
nvdsparsebbox_yolo_EXTERNAL_OBJECTS =

CMakeFiles/nvdsparsebbox_yolo.dir/cmake_device_link.o: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o
CMakeFiles/nvdsparsebbox_yolo.dir/cmake_device_link.o: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o
CMakeFiles/nvdsparsebbox_yolo.dir/cmake_device_link.o: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o
CMakeFiles/nvdsparsebbox_yolo.dir/cmake_device_link.o: CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o
CMakeFiles/nvdsparsebbox_yolo.dir/cmake_device_link.o: CMakeFiles/nvdsparsebbox_yolo.dir/build.make
CMakeFiles/nvdsparsebbox_yolo.dir/cmake_device_link.o: /usr/local/cuda/lib64/libcudart_static.a
CMakeFiles/nvdsparsebbox_yolo.dir/cmake_device_link.o: /usr/lib/aarch64-linux-gnu/librt.so
CMakeFiles/nvdsparsebbox_yolo.dir/cmake_device_link.o: CMakeFiles/nvdsparsebbox_yolo.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/longdd/Documents/ryolov5_jetson/deepstream/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CUDA device code CMakeFiles/nvdsparsebbox_yolo.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nvdsparsebbox_yolo.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nvdsparsebbox_yolo.dir/build: CMakeFiles/nvdsparsebbox_yolo.dir/cmake_device_link.o

.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/build

# Object files for target nvdsparsebbox_yolo
nvdsparsebbox_yolo_OBJECTS = \
"CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o" \
"CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o" \
"CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o" \
"CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o"

# External object files for target nvdsparsebbox_yolo
nvdsparsebbox_yolo_EXTERNAL_OBJECTS =

libnvdsparsebbox_yolo.so: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o
libnvdsparsebbox_yolo.so: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o
libnvdsparsebbox_yolo.so: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o
libnvdsparsebbox_yolo.so: CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o
libnvdsparsebbox_yolo.so: CMakeFiles/nvdsparsebbox_yolo.dir/build.make
libnvdsparsebbox_yolo.so: /usr/local/cuda/lib64/libcudart_static.a
libnvdsparsebbox_yolo.so: /usr/lib/aarch64-linux-gnu/librt.so
libnvdsparsebbox_yolo.so: CMakeFiles/nvdsparsebbox_yolo.dir/cmake_device_link.o
libnvdsparsebbox_yolo.so: CMakeFiles/nvdsparsebbox_yolo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/longdd/Documents/ryolov5_jetson/deepstream/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX shared library libnvdsparsebbox_yolo.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nvdsparsebbox_yolo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nvdsparsebbox_yolo.dir/build: libnvdsparsebbox_yolo.so

.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/build

CMakeFiles/nvdsparsebbox_yolo.dir/requires: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/decode_iou.cu.o.requires
CMakeFiles/nvdsparsebbox_yolo.dir/requires: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/csrc/cuda/nms_iou.cu.o.requires
CMakeFiles/nvdsparsebbox_yolo.dir/requires: CMakeFiles/nvdsparsebbox_yolo.dir/home/longdd/Documents/ryolov5_jetson/cpp/engine.cpp.o.requires
CMakeFiles/nvdsparsebbox_yolo.dir/requires: CMakeFiles/nvdsparsebbox_yolo.dir/nvdsparsebbox_yolo.cpp.o.requires

.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/requires

CMakeFiles/nvdsparsebbox_yolo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nvdsparsebbox_yolo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/clean

CMakeFiles/nvdsparsebbox_yolo.dir/depend:
	cd /home/longdd/Documents/ryolov5_jetson/deepstream/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/longdd/Documents/ryolov5_jetson/deepstream /home/longdd/Documents/ryolov5_jetson/deepstream /home/longdd/Documents/ryolov5_jetson/deepstream/build /home/longdd/Documents/ryolov5_jetson/deepstream/build /home/longdd/Documents/ryolov5_jetson/deepstream/build/CMakeFiles/nvdsparsebbox_yolo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nvdsparsebbox_yolo.dir/depend

