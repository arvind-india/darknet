DEBUG=1
GPU=1
EXEC=facedemo
VPATH=./
OBJDIR=./obj/

CXX=g++
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -L../../ -lm -pthread -ldarknet `pkg-config --libs opencv`
COMMON= -I../../include/
#CXXFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors -fno-stack-protector
CXXFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors
CXXFLAGS+= -Werror=implicit-function-declaration `pkg-config --cflags opencv`
#CXXFLAGS+= -I$(HOME)/anaconda2/include/opencv2/

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CXXFLAGS+= -DGPU
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CXXFLAGS+=$(OPTS)

OBJ= user_control.o setup.o demo_threads.o demo_main.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard ./*.h) Makefile ../../include/darknet.h

all: obj $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS) 

$(OBJDIR)%.o: %.cpp $(DEPS)
	@echo "Compiling $<"
	$(CXX) $(COMMON) $(CXXFLAGS) -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS)

