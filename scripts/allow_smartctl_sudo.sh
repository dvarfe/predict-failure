#!/usr/bin/env bash

# Usage: sudo ./set_smartctl_caps.sh [smartctl_path] [capabilities]
SMARTCTL_PATH=${1:-$(which smartctl || echo "/usr/sbin/smartctl")}
CAPS=${2:-'cap_sys_admin,cap_sys_rawio,cap_dac_read_search+ep'}

setcap "$CAPS" "$SMARTCTL_PATH" && \
  echo "Set $CAPS on $SMARTCTL_PATH" || \
  { echo "Failed to set capabilities on $SMARTCTL_PATH" >&2; exit 1; }