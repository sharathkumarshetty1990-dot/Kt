#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

python3 -m py_compile server.py editing_capabilities.py test_harness.py test_repairs.py test_api.py executor_smoke.py wsgi.py
python3 -m unittest test_repairs.py test_api.py
python3 executor_smoke.py
