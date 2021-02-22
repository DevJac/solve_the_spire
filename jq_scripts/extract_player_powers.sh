#!/usr/bin/env bash

jq '.sts_state.game_state.combat_state.player.powers[]?.id' log.txt | sort | uniq
