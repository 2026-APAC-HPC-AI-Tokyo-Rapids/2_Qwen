#!/usr/bin/env bash
# The SSH adapter and PBS non-interactive shells need the module initializer.
source /etc/profile.d/modules.sh
module load cuda/12.6.2 gcc/13.3.0
export CXX=/app/apps/gcc/13.3.0/bin/g++
