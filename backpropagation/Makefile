CPP=g++
CPPFLAGS= -g -Wall
OPTIMIZATION = -O3
CPPVERSION = -std=c++2a	# cpp20

# directories.
SRC_DIR = src
OBJ_DIR = objs
HEADERS_DIR = headers
BIN_DIR = bin

# extracting files.
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
HEADERS = $(wildcard $(HEADERS_DIR)/*.hpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))
BIN = $(BIN_DIR)/main

all: $(BIN)

$(BIN) : $(OBJS)
	$(CPP)	$(CPPFLAGS) $(OPTIMIZATION) $(CPPVERSION) -o $@ $(OBJS)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(HEADERS)
	$(CPP) $(CPPVERSION) -c $< -o $@

clean:
	rm -r $(OBJS)
	rm -r $(BIN)