# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zyi/Desktop/serialca

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zyi/Desktop/serialca/build

# Include any dependencies generated for this target.
include CMakeFiles/helios_serial.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/helios_serial.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/helios_serial.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/helios_serial.dir/flags.make

CMakeFiles/helios_serial.dir/src/Serial.cpp.o: CMakeFiles/helios_serial.dir/flags.make
CMakeFiles/helios_serial.dir/src/Serial.cpp.o: ../src/Serial.cpp
CMakeFiles/helios_serial.dir/src/Serial.cpp.o: CMakeFiles/helios_serial.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zyi/Desktop/serialca/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/helios_serial.dir/src/Serial.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/helios_serial.dir/src/Serial.cpp.o -MF CMakeFiles/helios_serial.dir/src/Serial.cpp.o.d -o CMakeFiles/helios_serial.dir/src/Serial.cpp.o -c /home/zyi/Desktop/serialca/src/Serial.cpp

CMakeFiles/helios_serial.dir/src/Serial.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/helios_serial.dir/src/Serial.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zyi/Desktop/serialca/src/Serial.cpp > CMakeFiles/helios_serial.dir/src/Serial.cpp.i

CMakeFiles/helios_serial.dir/src/Serial.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/helios_serial.dir/src/Serial.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zyi/Desktop/serialca/src/Serial.cpp -o CMakeFiles/helios_serial.dir/src/Serial.cpp.s

CMakeFiles/helios_serial.dir/src/CRC.cpp.o: CMakeFiles/helios_serial.dir/flags.make
CMakeFiles/helios_serial.dir/src/CRC.cpp.o: ../src/CRC.cpp
CMakeFiles/helios_serial.dir/src/CRC.cpp.o: CMakeFiles/helios_serial.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zyi/Desktop/serialca/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/helios_serial.dir/src/CRC.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/helios_serial.dir/src/CRC.cpp.o -MF CMakeFiles/helios_serial.dir/src/CRC.cpp.o.d -o CMakeFiles/helios_serial.dir/src/CRC.cpp.o -c /home/zyi/Desktop/serialca/src/CRC.cpp

CMakeFiles/helios_serial.dir/src/CRC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/helios_serial.dir/src/CRC.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zyi/Desktop/serialca/src/CRC.cpp > CMakeFiles/helios_serial.dir/src/CRC.cpp.i

CMakeFiles/helios_serial.dir/src/CRC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/helios_serial.dir/src/CRC.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zyi/Desktop/serialca/src/CRC.cpp -o CMakeFiles/helios_serial.dir/src/CRC.cpp.s

# Object files for target helios_serial
helios_serial_OBJECTS = \
"CMakeFiles/helios_serial.dir/src/Serial.cpp.o" \
"CMakeFiles/helios_serial.dir/src/CRC.cpp.o"

# External object files for target helios_serial
helios_serial_EXTERNAL_OBJECTS =

libhelios_serial.a: CMakeFiles/helios_serial.dir/src/Serial.cpp.o
libhelios_serial.a: CMakeFiles/helios_serial.dir/src/CRC.cpp.o
libhelios_serial.a: CMakeFiles/helios_serial.dir/build.make
libhelios_serial.a: CMakeFiles/helios_serial.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zyi/Desktop/serialca/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libhelios_serial.a"
	$(CMAKE_COMMAND) -P CMakeFiles/helios_serial.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/helios_serial.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/helios_serial.dir/build: libhelios_serial.a
.PHONY : CMakeFiles/helios_serial.dir/build

CMakeFiles/helios_serial.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/helios_serial.dir/cmake_clean.cmake
.PHONY : CMakeFiles/helios_serial.dir/clean

CMakeFiles/helios_serial.dir/depend:
	cd /home/zyi/Desktop/serialca/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zyi/Desktop/serialca /home/zyi/Desktop/serialca /home/zyi/Desktop/serialca/build /home/zyi/Desktop/serialca/build /home/zyi/Desktop/serialca/build/CMakeFiles/helios_serial.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/helios_serial.dir/depend

