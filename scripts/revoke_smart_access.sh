#!/usr/bin/env bash

sudo setcap -r /usr/sbin/smartctl && \
  echo "Removed capabilities from /usr/sbin/smartctl" || \
  { echo "Failed to remove capabilities from /usr/sbin/smartctl" >&2; exit 1; }