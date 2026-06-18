.PHONY: all julia python clean configure-julia configure-python

CMAKE ?= cmake
BUILD_ROOT ?= build
BUILD_TYPE ?= Release
MARBLE_GENERATE_PYI ?= ON

JULIA_CMAKE_DIR := $(BUILD_ROOT)/cmake/julia
PYTHON_CMAKE_DIR := $(BUILD_ROOT)/cmake/python
JULIA_LIB_DIR := $(BUILD_ROOT)/julia/lib
PYTHON_PACKAGE_DIR := $(BUILD_ROOT)/python/marble

all: julia python

julia: configure-julia
	$(CMAKE) --build $(JULIA_CMAKE_DIR) --target marble_julia

python: configure-python
	$(CMAKE) --build $(PYTHON_CMAKE_DIR) --target marble_python

configure-julia:
	$(CMAKE) -S . -B $(JULIA_CMAKE_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
		-DMARBLE_BUILD_JULIA=ON \
		-DMARBLE_BUILD_PYTHON=OFF \
		-DMARBLE_JULIA_LIB_DIR=$(CURDIR)/$(JULIA_LIB_DIR)

configure-python:
	$(CMAKE) -S . -B $(PYTHON_CMAKE_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
		-DMARBLE_BUILD_JULIA=OFF \
		-DMARBLE_BUILD_PYTHON=ON \
		-DMARBLE_GENERATE_PYI=$(MARBLE_GENERATE_PYI) \
		-DMARBLE_PYTHON_PACKAGE_DIR=$(CURDIR)/$(PYTHON_PACKAGE_DIR)

clean:
	rm -rf $(BUILD_ROOT)
	rm -rf python/build python/dist python/*.egg-info
	rm -f python/marble/_core*.so python/marble/_core*.pyd
	find python -type d -name __pycache__ -prune -exec rm -rf {} +
	find python -type f -name '*.pyc' -delete
