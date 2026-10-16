#!/usr/bin/env bash
set -eo pipefail

dagger_audit_project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$dagger_audit_project"
if [[ $# -gt 1 ]]; then
    echo "usage: sudo bash script/start_dagger_input_audit.sh [CAMERA_SDK_SERIAL]" >&2
    exit 2
fi
if pgrep -f '[/]hardware_elf3/hardware_elf3|[/]bxi_example_py_elf3/bxi_example_py_elf3_demo' >/dev/null; then
    echo "error: a hardware controller is already running; stop its owning launch before this audit." >&2
    echo "No process was stopped and no lock file was created." >&2
    exit 1
fi
exec python3 tools/diagnostics/dagger_inference_audit.py -- \
    bash script/start_dagger_hw_test.sh "$@"
