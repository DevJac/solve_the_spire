module STSAgents
using Utils

export GameData, DefaultGameData

const MAX_STS_HAND_SIZE = 10
const MAX_STS_MONSTER_COUNT = 5

struct GameData
    card_ids
    potion_ids
    player_power_ids
    monster_power_ids
end

const DefaultGameData = GameData(
    readlines("game_data/card_ids.txt"),
    readlines("game_data/potion_ids.txt"),
    readlines("game_data/player_power_ids.txt"),
    readlines("game_data/monster_power_ids.txt"))

function find_p1(needle, haystack)
    i = find(needle, haystack)
    isnothing(i) ? length(haystack)+1 : i
end

function encode_card_in_hand(game_data, card_json)
    encoding = zeros(Float32, length(game_data.card_ids)+4)
    i = find_p1(card_json["id"], game_data.card_ids)
    encoding[i] = 1
    encoding[end-2] = 1
    encoding[end-1] = card_json["upgrades"]
    encoding[end] = card_json["cost"]
    Float32.(encoding)
end

function encode_cards_in_draw_discard_pile(game_data, cards_json)
    encoding = zeros(Float32, (length(game_data.card_ids)+1)*2)
    for cj in cards_json
        i = find_p1(cj["id"], game_data.card_ids)
        encoding[i*2-1] += 1
        encoding[i*2] += cj["upgrades"]
    end
    for i in 1:length(game_data.card_ids)+1
        encoding[i*2] /= encoding[i*2-1]
        encoding[i*2-1] /= length(cards_json)
    end
    Float32.(encoding)
end

function encode_potion(game_data, potion_json)
    encoding = zeros(Float32, length(game_data.potion_ids))
    if potion_json["id"] != "Potion Slot"
        encoding[find(potion_json["id"], game_data.potion_ids)] = 1
    end
    Float32.(encoding)
end

function encode_player_powers(game_data, player_powers_json)
    encoding = zeros(Float32, (length(game_data.player_power_ids)+1)*2)
    for pj in player_powers_json
        i = find_p1(pj["id"], game_data.player_power_ids)
        encoding[i*2-1] = 1
        encoding[i*2] = pj["amount"]
    end
    Float32.(encoding)
end

function encode_player(player_json, monsters_json)
    encoding = zeros(Float32, 5)
    encoding[1] = player_json["current_hp"]
    encoding[2] = player_json["max_hp"]
    encoding[3] = player_json["current_hp"] / player_json["max_hp"]
    encoding[4] = player_json["block"]
    encoding[5] = player_json["block"] - sum(monster_total_attack, monsters_json)
    Float32.(encoding)
end

function encode_monster_powers(game_data, monster_powers_json)
    encoding = zeros(Float32, (length(game_data.monster_power_ids)+1)*2)
    for pj in monster_powers_json
        i = find_p1(pj["id"], game_data.monster_power_ids)
        encoding[i] = 1
        encoding[i*2] = pj["amount"]
    end
    Float32.(encoding)
end

function monster_total_attack(monster_json)
    if !contains(monster_json["intent"], "ATTACK")
        0
    else
        monster_json["move_adjusted_damage"] * monster_json["move_hits"]
    end
end

function encode_monster(monster_json)
    encoding = zeros(Float32, 7)
    encoding[1] = monster_json["move_hits"]
    encoding[2] = monster_json["move_adjusted_damage"] * monster_json["move_hits"]
    encoding[3] = monster_json["block"]
    encoding[4] = monster_json["current_hp"]
    encoding[5] = monster_json["max_hp"]
    encoding[6] = monster_json["current_hp"] / monster_json["max_hp"]
    Float32.(encoding)
end

function encode(game_state_json)
    encodings = []
end

end # module
