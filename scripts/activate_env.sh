#!/bin/bash

# Activate project virtual environment
source .venv/bin/activate

# Locate NVIDIA cuSolver library
CUSOLVER_LIB=$(
    python -c 'import nvidia.cusolver, pathlib; print(pathlib.Path(next(iter(nvidia.cusolver.__path__))) / "lib")'
)

# Configure CUDA library paths
export LD_LIBRARY_PATH="${CUSOLVER_LIB}:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "ArogyaNet environment activated."
echo "Python: $(which python)"
echo "cuSolver: ${CUSOLVER_LIB}"