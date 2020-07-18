#!/usr/bin/env bash

set -euo pipefail
modules="test espnet_model_zoo setup.py"
# black
if ! black --check ${modules}; then
    printf 'Please apply:\n    $ black %s\n' "${modules}"
    exit 1
fi
# flake8
flake8 --show-source $modules
# pycodestyle
pycodestyle -r $modules --show-source --show-pep8 
pytest -q
