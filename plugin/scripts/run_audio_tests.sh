#!/usr/bin/env bash
# Holy Shifter — objective audio test harness runner.
# Creates/uses an isolated venv (outside iCloud), installs pedalboard+numpy on
# first run, then runs audio_test_harness.py against the VST3.
#
# Usage:  bash run_audio_tests.sh [path/to/Plugin.vst3]
set -euo pipefail

VENV="$HOME/.cache/holyshifter-audiotest/venv"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$VENV/bin/python" ]; then
  echo "First run: creating venv + installing pedalboard (one-time)…"
  mkdir -p "$(dirname "$VENV")"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q pedalboard numpy
fi

exec "$VENV/bin/python" "$HERE/audio_test_harness.py" "$@"
