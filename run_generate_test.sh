#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/src" && python sample.py --gpu 0
