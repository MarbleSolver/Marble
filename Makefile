.PHONY: all python julia clean

all:
	cmake --preset all
	cmake --build --preset all --parallel

python:
	cmake --preset python
	cmake --build --preset python --parallel

julia:
	cmake --preset julia
	cmake --build --preset julia --parallel

clean:
	rm -rf build
