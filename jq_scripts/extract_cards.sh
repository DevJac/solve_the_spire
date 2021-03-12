#!/usr/bin/env bash

jq -r '(.sts_state.game_state.combat_state.hand[]?.id),(.sts_state.game_state.screen_state.cards[]?.id)' log.txt | sort | uniq
