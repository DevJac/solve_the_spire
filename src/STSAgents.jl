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

export Encoder, encode

struct Encoder
    encoders :: Vector{Function}
end
Encoder() = Encoder([])

function add_encoder(f, encoder::Encoder)
    push!(encoder.encoders, f)
end

function encode(encoder::Encoder, sts_state_json)
    encoded = map(encoder.encoders) do e
        e(sts_state_json)
    end
    Float32.(encoded)
end

function make_hand_card_encoder(game_data)
    encoder = Encoder()
    ae(f) = add_encoder(f, encoder)
    for card_id in game_data.card_ids
        ae() do j
            j["id"] == card_id
        end
    end
    ae() do j
        j["upgrades"]
    end
    ae() do j
        j["cost"]
    end
    encoder
end

function make_draw_discard_encoder(game_data)
    encoder = Encoder()
    ae(f) = add_encoder(f, encoder)
    draw(j) = j["game_state"]["combat_state"]["draw_pile"]
    discard(j) = j["game_state"]["combat_state"]["discard_pile"]
    for pile in (draw, discard)
        for card_id in game_data.card_ids
            ae() do j
                count(c -> c["id"] == card_id, pile(j)) / length(pile(j))
            end
            ae() do j
                matching = filter(c -> c["id"] == card_id, pile(j))
                sum(c -> c["upgrades"], matching) / length(matching)
            end
        end
    end
    encoder
end

function make_player_powers_encoder(game_data)
    encoder = Encoder()
    ae(f) = add_encoder(f, encoder)
    powers(j) = j["game_state"]["combat_state"]["player"]["powers"]
    for power_id in game_data.player_power_ids
        ae() do j
            any(powers(j)) do power
                power["id"] == power_id
            end
        end
        ae() do j
            matching = filter(p -> p["id"] == power_id, powers(j))
            !isempty(matching) ? only(matching)["amount"] : 0
        end
    end
    encoder
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
