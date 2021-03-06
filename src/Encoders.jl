module Encoders
using Memoize
using StatsBase
using Zygote

export GameData, DefaultGameData

struct GameData
    boss_ids
    card_ids
    card_rarities
    card_types
    monster_ids
    monster_power_ids
    player_power_ids
    potion_ids
    relic_ids
    screen_types
end

const DefaultGameData = GameData(
    readlines("game_data/boss_ids.txt"),
    readlines("game_data/card_ids.txt"),
    readlines("game_data/card_rarities.txt"),
    readlines("game_data/card_types.txt"),
    readlines("game_data/monster_ids.txt"),
    readlines("game_data/monster_power_ids.txt"),
    readlines("game_data/player_power_ids.txt"),
    readlines("game_data/potion_ids.txt"),
    readlines("game_data/relic_ids.txt"),
    readlines("game_data/screen_types.txt"))

struct Encoder
    name     :: String
    precoder :: Function
    encoders :: Vector{Function}
end
Encoder(name) = Encoder(name, identity, [])
Encoder(name, precoder) = Encoder(name, precoder, [])

(encoder::Encoder)(sts_state_json, args...; kwargs...) = Zygote.@ignore encode(encoder, sts_state_json, args...; kwargs...)

@memoize function encode(encoder::Encoder, sts_state_json, args...; kwargs...)
    encoded = map(encoder.encoders) do e
        e(encoder.precoder(sts_state_json, args...; kwargs...))
    end
    Float32.(encoded)
end

Base.show(io::IO, encoder::Encoder) = println(io, "<Encoder: $(encoder.name) $(length(encoder))>")

Base.length(e::Encoder) = length(e.encoders)

function add_encoder(f, encoder::Encoder)
    push!(encoder.encoders, f)
end

export make_screen_type_encoder, make_card_encoder, make_player_basic_encoder, make_player_combat_encoder
export make_monster_encoder, make_relics_encoder, make_potions_encoder, make_map_encoder

function make_screen_type_encoder(game_data)
    encoder = Encoder("Screen Type")
    ae(f) = add_encoder(f, encoder)
    for screen_type in game_data.screen_types
        ae() do j
            @assert j["game_state"]["screen_type"] in game_data.screen_types
            j["game_state"]["screen_type"] == screen_type
        end
    end
    encoder
end

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
    ae() do j
        j["game_state"]["gold"]
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
        j["game_state"]["combat_state"]["cards_discarded_this_turn"]
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
        get(j, "move_adjusted_damage", 0)
    end
    ae() do j
        get(j, "move_hits", 0)
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
            !isempty(matching) ? only(matching)["counter"] : 0
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

function make_map_encoder(game_data)
    truncate(len) = x -> x[1:min(len, length(x))]
    function precoder(j, x, y)
        mps = map_paths(j["game_state"]["map"], x, y)
        one_path_counts = map_path_counts(map(truncate(1), mps))
        two_path_counts = map_path_counts(map(truncate(2), mps))
        full_path_counts = map_path_counts(mps)
        (j, one_path_counts, two_path_counts, full_path_counts)
    end
    encoder = Encoder("Map", precoder)
    ae(f) = add_encoder(f, encoder)
    for i in 1:length(MAP_ROOM_TYPES)
        ae() do d
            d[2][i][1]
        end
        ae() do d
            d[3][i][1]
        end
        ae() do d
            d[3][i][2]
        end
        ae() do d
            d[4][i][1]
        end
        ae() do d
            d[4][i][2]
        end
    end
    for boss_id in game_data.boss_ids
        ae() do d
            d[1]["game_state"]["act_boss"] == boss_id
        end
    end
    ae() do d
        d[1]["game_state"]["floor"]
    end
    encoder
end

function map_path_counts(map_paths)
    path_counts = StatsBase.countmap.(map_paths)
    map(MAP_ROOM_TYPES) do room_type
        (minimum(pc -> get(pc, room_type, 0), path_counts), maximum(pc -> get(pc, room_type, 0), path_counts))
    end
end

function map_paths(map_state, x, y)
    symbol_map = Dict{Tuple{Int8,Int8},Char}()
    child_map = Dict{Tuple{Int8,Int8},Vector{Tuple{Int8,Int8}}}()
    for node in map_state
        coord = (node["x"], node["y"])
        symbol_map[coord] = only(node["symbol"])
        for child in node["children"]
            child_map[coord] = push!(get(child_map, coord, []), (child["x"], child["y"]))
        end
    end
    map_paths′(symbol_map, child_map, Int8(x), Int8(y), Vector{Char}[], Char[])
end

function map_paths′(
        symbol_map::Dict{Tuple{Int8,Int8},Char},
        child_map::Dict{Tuple{Int8,Int8},Vector{Tuple{Int8,Int8}}},
        x::Int8,
        y::Int8,
        all_paths::Vector{Vector{Char}},
        path::Vector{Char})
    leaf_node = true
    if (x, y) in keys(symbol_map); push!(path, symbol_map[(x, y)]) end
    for child_node in get(child_map, (x, y), [])
        leaf_node = false
        map_paths′(symbol_map, child_map, child_node[1], child_node[2], all_paths, copy(path))
    end
    if leaf_node
        push!(all_paths, copy(path))
    end
    all_paths
end

export screen_type_encoder, card_encoder, player_basic_encoder, player_combat_encoder
export monster_encoder, relics_encoder, potions_encoder, map_encoder

const screen_type_encoder = make_screen_type_encoder(DefaultGameData)
const card_encoder = make_card_encoder(DefaultGameData)
const player_basic_encoder = make_player_basic_encoder()
const player_combat_encoder = make_player_combat_encoder(DefaultGameData)
const monster_encoder = make_monster_encoder(DefaultGameData)
const relics_encoder = make_relics_encoder(DefaultGameData)
const potions_encoder = make_potions_encoder(DefaultGameData)
const map_encoder = make_map_encoder(DefaultGameData)

end # module
