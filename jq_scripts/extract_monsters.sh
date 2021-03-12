#!/usr/bin/env bash

jq -r '.. | .monsters?[]?.id' log.txt | sort | uniq
