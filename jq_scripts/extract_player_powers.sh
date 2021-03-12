#!/usr/bin/env bash

jq -r '.sts_state.game_state.combat_state.player.powers[]?.id' log.txt | sort | uniq
