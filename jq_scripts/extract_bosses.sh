#!/usr/bin/env bash

jq -r '.. | .act_boss? | select(. != null)' log.txt | sort | uniq
