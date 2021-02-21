#!/usr/bin/env bash

jq --tab -C '.sts_state.game_state.combat_state.monsters | arrays | .[] | objects | .powers[].id' log.txt | sort | uniq
