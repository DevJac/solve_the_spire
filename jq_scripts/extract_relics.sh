#!/usr/bin/env bash

jq -r '.. | .relics?[]?.id' log.txt | sort | uniq
