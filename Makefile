LDFLAGS = -ldlib -lpthread -ljpeg -lpng -lX11 -lopenblas -lsqlite3 `pkg-config --cflags --libs opencv`


# Define the target executable name
TARGET = main

# Define the source files
SRC = main_pi.cc face_features.cc face_features.hh db_utils.cc db_utils.hh

# Define the output directories for object and binary files
OBJ_DIR = obj
BIN_DIR = bin

# Automatically generate object file names
OBJ = $(SRC:%.cpp=$(OBJ_DIR)/%.o)

# Default target to compile the executable
all: $(BIN_DIR)/$(TARGET)

# Link the executable
$(BIN_DIR)/$(TARGET): $(OBJ)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Compile source files to object files
$(OBJ_DIR)/%.o: %.cc
	mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean
