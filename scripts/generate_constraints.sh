#!/bin/sh
set -xe

rm -rf venv
python -m venv venv
. venv/bin/activate

export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
export PIP_CONSTRAINT=./constraints-dev.txt
PIP_INSTALL="pip install --no-cache-dir --constraint $PIP_CONSTRAINT --upgrade"

$PIP_INSTALL --upgrade pip
$PIP_INSTALL pip-tools # unpinned; ironic?
$PIP_INSTALL scikit_build_core torch

UNSAFE_PACKAGES=(
    "flash-attn"
    "llama_cpp_python"
)

for package in "${UNSAFE_PACKAGES[@]}"; do
    $PIP_INSTALL --no-deps --force-reinstall "$package" -c constraints-dev.txt
done

pip-compile --no-build-isolation --unsafe-package flash-attn --unsafe-package llama_cpp_python --verbose --all-extras -U -r --no-annotate --no-emit-options --generate-hashes --reuse-hashes --no-header --output-file=constraints-dev.txt -c constraints-dev.txt.in requirements*.txt requirements/cuda.txt
sed 's/\[.*\]//' -i constraints-dev.txt

# TODO: remove after constraint is moved from tox.ini to constraints-dev.txt
sed '/^isort==/d' -i constraints-dev.txt
