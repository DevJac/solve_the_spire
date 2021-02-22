#!/usr/bin/env bash

jq '.. | .relics?[]?.id' log.txt | sort | uniq
