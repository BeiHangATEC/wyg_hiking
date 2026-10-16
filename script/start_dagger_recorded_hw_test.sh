#!/usr/bin/env bash
set -eo pipefail

dagger_record_project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Compatibility entrypoint; local recording is independent of hardware startup.
exec bash "$dagger_record_project/script/start_dagger_hw_test.sh" "$@"
