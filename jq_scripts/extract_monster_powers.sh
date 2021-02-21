#!/usr/bin/env bash

jq '.sts_state.game_state.combat_state.monsters | arrays | .[] | objects | .powers[].id' log.txt | sort | uniq
