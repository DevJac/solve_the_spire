#!/usr/bin/env bash

jq --tab -C 'del(.sts_state.game_state.map) | .sts_state.game_state.combat_state | (.monsters|arrays|.[]),.player|objects|.powers[].id' log.txt | sort | uniq
