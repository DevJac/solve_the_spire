module Encoders
using StatsBase
using Zygote

export GameData, DefaultGameData

struct GameData
    card_ids
    card_rarities
    card_types
    monster_ids
    monster_power_ids
    player_power_ids
    potion_ids
    relic_ids
end

const DefaultGameData = GameData(
    readlines("game_data/card_ids.txt"),
    readlines("game_data/card_rarities.txt"),
    readlines("game_data/card_types.txt"),
    readlines("game_data/monster_ids.txt"),
    readlines("game_data/monster_power_ids.txt"),
    readlines("game_data/player_power_ids.txt"),
    readlines("game_data/potion_ids.txt"),
    readlines("game_data/relic_ids.txt"))

struct Encoder
    name     :: String
    precoder :: Function
    encoders :: Vector{Function}
end
Encoder(name) = Encoder(name, identity, [])
Encoder(name, precoder) = Encoder(name, precoder, [])

function (encoder::Encoder)(sts_state_json, args...; kwargs...)
    Zygote.ignore() do
        encoded = map(encoder.encoders) do e
            e(encoder.precoder(sts_state_json, args...; kwargs...))
        end
        Float32.(encoded)
    end
end

Base.show(io::IO, encoder::Encoder) = println(io, "<Encoder: $(encoder.name) $(length(encoder))>")

Base.length(e::Encoder) = length(e.encoders)

function add_encoder(f, encoder::Encoder)
    push!(encoder.encoders, f)
end

export make_card_encoder, make_player_basic_encoder, make_player_combat_encoder
export make_monster_encoder, make_relics_encoder, make_potions_encoder, make_map_encoder

function make_card_encoder(game_data)
    encoder = Encoder("Card")
    ae(f) = add_encoder(f, encoder)
    for card_id in game_data.card_ids
        ae() do j
            j["id"] == card_id
        end
    end
    for card_rarity in game_data.card_rarities
        ae() do j
            j["rarity"] == card_rarity
        end
    end
    for card_type in game_data.card_types
        ae() do j
            j["type"] == card_type
        end
    end
    ae() do j
        j["has_target"]
    end
    ae() do j
        j["cost"]
    end
    ae() do j
        j["upgrades"]
    end
    encoder
end

function make_player_basic_encoder()
    encoder = Encoder("Player Basic")
    ae(f) = add_encoder(f, encoder)
    ae() do j
        j["game_state"]["current_hp"]
    end
    ae() do j
        j["game_state"]["max_hp"]
    end
    ae() do j
        j["game_state"]["current_hp"] / j["game_state"]["max_hp"]
    end
    encoder
end

function make_player_combat_encoder(game_data)
    encoder = Encoder("Player Combat")
    ae(f) = add_encoder(f, encoder)
    player(j) = j["game_state"]["combat_state"]["player"]
    powers(j) = player(j)["powers"]
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
    ae() do j
        player(j)["current_hp"]
    end
    ae() do j
        player(j)["max_hp"]
    end
    ae() do j
        player(j)["current_hp"] / player(j)["max_hp"]
    end
    ae() do j
        player(j)["energy"]
    end
    ae() do j
        player(j)["block"]
    end
    ae() do j
        player(j)["block"] - sum(monster_total_attack, j["game_state"]["combat_state"]["monsters"])
    end
    encoder
end

function monster_total_attack(monster_json)
    if !contains(monster_json["intent"], "ATTACK")
        0
    else
        monster_json["move_adjusted_damage"] * monster_json["move_hits"]
    end
end

function make_monster_encoder(game_data)
    encoder = Encoder("Monster")
    ae(f) = add_encoder(f, encoder)
    for monster_id in game_data.monster_ids
        ae() do j
            j["id"] == monster_id
        end
    end
    for power_id in game_data.monster_power_ids
        ae() do j
            any(j["powers"]) do power
                power["id"] == power_id
            end
        end
        ae() do j
            matching = filter(p -> p["id"] == power_id, j["powers"])
            !isempty(matching) ? only(matching)["amount"] : 0
        end
    end
    ae() do j
        j["current_hp"]
    end
    ae() do j
        j["max_hp"]
    end
    ae() do j
        j["current_hp"] / j["max_hp"]
    end
    ae() do j
        j["block"]
    end
    ae() do j
        j["move_adjusted_damage"]
    end
    ae() do j
        j["move_hits"]
    end
    ae() do j
        monster_total_attack(j)
    end
    encoder
end

function make_relics_encoder(game_data)
    encoder = Encoder("Relic")
    ae(f) = add_encoder(f, encoder)
    for relic_id in game_data.relic_ids
        ae() do j
            any(r -> r["id"] == relic_id, j)
        end
        ae() do j
            matching = filter(r -> r["id"] == relic_id, j)
            !isempty(matching) ? only(matching)["amount"] : 0
        end
    end
    encoder
end

function make_potions_encoder(game_data)
    encoder = Encoder("Potion")
    ae(f) = add_encoder(f, encoder)
    for potion_id in game_data.potion_ids
        ae() do j
            count(p -> p["id"] == potion_id, j)
        end
    end
    encoder
end

const MAP_ROOM_TYPES = ('T', 'E', 'R', 'M', '$', '?')

function make_map_encoder()
    truncate(len) = x -> x[1:min(len, length(x))]
    function precoder(j, x, y)
        one_path_counts = map_path_counts(map(truncate(1), map_paths(j, x, y)))
        two_path_counts = map_path_counts(map(truncate(2), map_paths(j, x, y)))
        full_path_counts = map_path_counts(map_paths(j, x, y))
        (one_path_counts, two_path_counts, full_path_counts)
    end
    encoder = Encoder("Map", precoder)
    ae(f) = add_encoder(f, encoder)
    for i in 1:length(MAP_ROOM_TYPES)
        ae() do d
            d[1][i][1]
        end
        ae() do d
            d[2][i][1]
        end
        ae() do d
            d[2][i][2]
        end
        ae() do d
            d[3][i][1]
        end
        ae() do d
            d[3][i][2]
        end
    end
    encoder
end

function map_path_counts(map_paths)
    path_counts = StatsBase.countmap.(map_paths)
    map(MAP_ROOM_TYPES) do room_type
        (minimum(pc -> get(pc, room_type, 0), path_counts), maximum(pc -> get(pc, room_type, 0), path_counts))
    end
end

function map_paths(map_state, x, y, all_paths=Vector{Char}[], path=Char[])
    leaf_node = true
    for map_node in map_state
        if map_node["x"] == x && map_node["y"] == y
            push!(path, only(map_node["symbol"]))
            for child_node in map_node["children"]
                leaf_node = false
                map_paths(map_state, child_node["x"], child_node["y"], all_paths, deepcopy(path))
            end
            break
        end
    end
    if leaf_node
        push!(all_paths, deepcopy(path))
    end
    all_paths
end

export card_encoder, player_basic_encoder, player_combat_encoder
export monster_encoder, relics_encoder, potions_encoder, map_encoder

const card_encoder = make_card_encoder(DefaultGameData)
const player_basic_encoder = make_player_basic_encoder()
const player_combat_encoder = make_player_combat_encoder(DefaultGameData)
const monster_encoder = make_monster_encoder(DefaultGameData)
const relics_encoder = make_relics_encoder(DefaultGameData)
const potions_encoder = make_potions_encoder(DefaultGameData)
const map_encoder = make_map_encoder()

end # module
