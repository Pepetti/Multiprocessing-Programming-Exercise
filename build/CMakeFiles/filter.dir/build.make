# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.9

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = F:\Kouluhommat\mpsp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = F:\Kouluhommat\mpsp\build

# Include any dependencies generated for this target.
include CMakeFiles/filter.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/filter.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/filter.dir/flags.make

CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj: CMakeFiles/filter.dir/flags.make
CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj: CMakeFiles/filter.dir/includes_CXX.rsp
CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj: ../code_opencl/lodepng.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=F:\Kouluhommat\mpsp\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj"
	C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\filter.dir\code_opencl\lodepng.cpp.obj -c F:\Kouluhommat\mpsp\code_opencl\lodepng.cpp

CMakeFiles/filter.dir/code_opencl/lodepng.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/filter.dir/code_opencl/lodepng.cpp.i"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E F:\Kouluhommat\mpsp\code_opencl\lodepng.cpp > CMakeFiles\filter.dir\code_opencl\lodepng.cpp.i

CMakeFiles/filter.dir/code_opencl/lodepng.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/filter.dir/code_opencl/lodepng.cpp.s"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S F:\Kouluhommat\mpsp\code_opencl\lodepng.cpp -o CMakeFiles\filter.dir\code_opencl\lodepng.cpp.s

CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj.requires:

.PHONY : CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj.requires

CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj.provides: CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj.requires
	$(MAKE) -f CMakeFiles\filter.dir\build.make CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj.provides.build
.PHONY : CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj.provides

CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj.provides.build: CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj


CMakeFiles/filter.dir/code_opencl/main.cpp.obj: CMakeFiles/filter.dir/flags.make
CMakeFiles/filter.dir/code_opencl/main.cpp.obj: CMakeFiles/filter.dir/includes_CXX.rsp
CMakeFiles/filter.dir/code_opencl/main.cpp.obj: ../code_opencl/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=F:\Kouluhommat\mpsp\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/filter.dir/code_opencl/main.cpp.obj"
	C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\filter.dir\code_opencl\main.cpp.obj -c F:\Kouluhommat\mpsp\code_opencl\main.cpp

CMakeFiles/filter.dir/code_opencl/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/filter.dir/code_opencl/main.cpp.i"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E F:\Kouluhommat\mpsp\code_opencl\main.cpp > CMakeFiles\filter.dir\code_opencl\main.cpp.i

CMakeFiles/filter.dir/code_opencl/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/filter.dir/code_opencl/main.cpp.s"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S F:\Kouluhommat\mpsp\code_opencl\main.cpp -o CMakeFiles\filter.dir\code_opencl\main.cpp.s

CMakeFiles/filter.dir/code_opencl/main.cpp.obj.requires:

.PHONY : CMakeFiles/filter.dir/code_opencl/main.cpp.obj.requires

CMakeFiles/filter.dir/code_opencl/main.cpp.obj.provides: CMakeFiles/filter.dir/code_opencl/main.cpp.obj.requires
	$(MAKE) -f CMakeFiles\filter.dir\build.make CMakeFiles/filter.dir/code_opencl/main.cpp.obj.provides.build
.PHONY : CMakeFiles/filter.dir/code_opencl/main.cpp.obj.provides

CMakeFiles/filter.dir/code_opencl/main.cpp.obj.provides.build: CMakeFiles/filter.dir/code_opencl/main.cpp.obj


# Object files for target filter
filter_OBJECTS = \
"CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj" \
"CMakeFiles/filter.dir/code_opencl/main.cpp.obj"

# External object files for target filter
filter_EXTERNAL_OBJECTS =

filter.exe: CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj
filter.exe: CMakeFiles/filter.dir/code_opencl/main.cpp.obj
filter.exe: CMakeFiles/filter.dir/build.make
filter.exe: C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v9.2/lib/Win32/OpenCL.lib
filter.exe: CMakeFiles/filter.dir/linklibs.rsp
filter.exe: CMakeFiles/filter.dir/objects1.rsp
filter.exe: CMakeFiles/filter.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=F:\Kouluhommat\mpsp\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable filter.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\filter.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/filter.dir/build: filter.exe

.PHONY : CMakeFiles/filter.dir/build

CMakeFiles/filter.dir/requires: CMakeFiles/filter.dir/code_opencl/lodepng.cpp.obj.requires
CMakeFiles/filter.dir/requires: CMakeFiles/filter.dir/code_opencl/main.cpp.obj.requires

.PHONY : CMakeFiles/filter.dir/requires

CMakeFiles/filter.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\filter.dir\cmake_clean.cmake
.PHONY : CMakeFiles/filter.dir/clean

CMakeFiles/filter.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" F:\Kouluhommat\mpsp F:\Kouluhommat\mpsp F:\Kouluhommat\mpsp\build F:\Kouluhommat\mpsp\build F:\Kouluhommat\mpsp\build\CMakeFiles\filter.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/filter.dir/depend
