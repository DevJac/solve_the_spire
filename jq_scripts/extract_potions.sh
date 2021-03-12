#!/usr/bin/env bash

jq -r '.. | .potions?[]? | select(.id != "Potion Slot") | .id' log.txt | sort | uniq
