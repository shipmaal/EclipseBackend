# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -g -Wall

# Target executable
TARGET = myprogram

# Source files
SRCS = example.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<  -o $@

clean:
	$(RM) $(OBJS) $(TARGET)