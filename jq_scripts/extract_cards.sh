#!/usr/bin/env bash

jq '(.sts_state.game_state.combat_state.hand | arrays | .[].id),(.sts_state.game_state.screen_state.cards | arrays | .[].id)' log.txt | sort | uniq
