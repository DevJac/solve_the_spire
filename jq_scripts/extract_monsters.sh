#!/usr/bin/env bash

jq '.. | .monsters?[]?.id' log.txt | sort | uniq
