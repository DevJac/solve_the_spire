#!/usr/bin/env bash

jq '.. | .potions?[]?.id' log.txt | sort | uniq
