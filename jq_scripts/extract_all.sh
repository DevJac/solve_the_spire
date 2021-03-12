#!/usr/bin/env bash

./jq_scripts/extract_cards.sh > game_data/card_ids.txt
./jq_scripts/extract_monster_powers.sh > game_data/monster_power_ids.txt
./jq_scripts/extract_monsters.sh > game_data/monster_ids.txt
./jq_scripts/extract_player_powers.sh > game_data/player_power_ids.txt
./jq_scripts/extract_potions.sh > game_data/potion_ids.txt
./jq_scripts/extract_relics.sh > game_data/relic_ids.txt
