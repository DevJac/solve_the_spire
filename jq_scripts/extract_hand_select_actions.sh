#!/usr/bin/env bash

jq -r '. | del(.sts_state.game_state.map) | select(.sts_state.game_state.screen_type == "HAND_SELECT") | .sts_state.game_state.current_action' log.txt | sort -u
