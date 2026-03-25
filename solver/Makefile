CPP      = g++

# External libraries
PKG_CONFIG ?= pkg-config

# yaml-cpp (required)
YAML_CPP_CFLAGS  := $(shell $(PKG_CONFIG) --cflags yaml-cpp 2>/dev/null)
YAML_CPP_LDFLAGS := $(shell $(PKG_CONFIG) --libs   yaml-cpp 2>/dev/null)

ifeq ($(YAML_CPP_LDFLAGS),)
$(error yaml-cpp not found. Please install yaml-cpp development package)
endif

#CFLAGS   = -Wall -pthread -fopenmp -O3 -std=c++17 # Uncomment to optimize
CFLAGS   = -Wall -pthread -fopenmp -std=c++17 $(YAML_CPP_CFLAGS)
LDFLAGS  = -lgsl -lgslcblas -lm $(YAML_CPP_LDFLAGS)

SRCDIR   = src
OBJDIR   = obj
BINDIR   = bin
INCDIR   = include

# Find all entry points (*_main.cpp)
MAIN_SOURCES = $(wildcard $(SRCDIR)/*_main.cpp)
TARGETS      = $(patsubst $(SRCDIR)/%_main.cpp,$(BINDIR)/%,$(MAIN_SOURCES))

# Common sources = everything except *_main.cpp
COMMON_SOURCES = $(filter-out $(MAIN_SOURCES),$(shell find $(SRCDIR) -name '*.cpp'))
COMMON_OBJECTS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(COMMON_SOURCES))

HEADERS  = $(wildcard $(INCDIR)/*.hpp)
INCLUDES = -I$(INCDIR)

ifdef VERBOSE
    CFLAGS += -DVERBOSE=true
endif

all: $(TARGETS)

# Generic rule: each binary depends on common objects + its own main
$(BINDIR)/%: $(COMMON_OBJECTS) $(OBJDIR)/%_main.o
	@mkdir -p $(@D)
	@echo "Linking $@..."
	$(CPP) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)
	@mkdir -p $(BINDIR)/assets
	cp assets/*_manual.txt $(BINDIR)/assets/
	@echo "$@ built successfully"

# Compile rule for all .cpp files with automatic dependency generation
OBJFILES = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(COMMON_SOURCES) $(MAIN_SOURCES))

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling $<..."
	$(CPP) $(CFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@
	@echo "$< compiled successfully"

# Include automatically generated dependencies
DEPFILES = $(OBJFILES:.o=.d)
-include $(DEPFILES)

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean