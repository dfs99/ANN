CPP=g++
CPPFLAGS= -g -Wall
OPTIMIZATION = -O3
CPPVERSION = -std=c++2a	# cpp20

# backpropagation
# bckprop
srcs= bckprop.cpp backpropagation.cpp
objs= bckprop.o backpropagation.o

all: $(objs)

.cpp.o:
	$(CPP) $(CPPFLAGS) $(OPTIMIZATION) $(CPPVERSION) $< -o $@

clean:
	rm -rf *.o

# check memory errors and leaks with Valgrind.
memory: $(objs)
	valgrind --tool=memcheck --leak-check=full ./bckprop.o
	valgrind --tool=memcheck --leak-check=full ./backpropagation.o
