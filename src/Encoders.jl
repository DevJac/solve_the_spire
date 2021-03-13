module Encoders
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
    name :: String
    encoders :: Vector{Function}
end
Encoder(name) = Encoder(name, [])

function (encoder::Encoder)(sts_state_json)
    Zygote.ignore() do
        encoded = map(encoder.encoders) do e
            e(sts_state_json)
        end
        Float32.(encoded)
    end
end

Base.show(io::IO, encoder::Encoder) = println(io, "<Encoder: $name $(length(encoder))>")

Base.length(e::Encoder) = length(e.encoders)

function add_encoder(f, encoder::Encoder)
    push!(encoder.encoders, f)
end

export make_hand_card_encoder, make_draw_discard_encoder, make_player_encoder, make_monsters_encoder

function make_hand_card_encoder(game_data)
    encoder = Encoder("Hand Card")
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

function make_draw_discard_encoder(game_data)
    encoder = Encoder("Draw Discard")
    ae(f) = add_encoder(f, encoder)
    draw(j) = j["game_state"]["combat_state"]["draw_pile"]
    discard(j) = j["game_state"]["combat_state"]["discard_pile"]
    for pile in (draw, discard)
        for card_id in game_data.card_ids
            ae() do j
                isempty(pile(j)) ? 0 : count(c -> c["id"] == card_id, pile(j)) / length(pile(j))
            end
            ae() do j
                matching = filter(c -> c["id"] == card_id, pile(j))
                isempty(matching) ? 0 : sum(c -> c["upgrades"], matching) / length(matching)
            end
        end
    end
    encoder
end

function make_player_encoder(game_data)
    encoder = Encoder("Player")
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

function make_monsters_encoder(game_data)
    encoder = Encoder("Monsters")
    ae(f) = add_encoder(f, encoder)
    monsters(j) = j["game_state"]["combat_state"]["monsters"]
    for monster in game_data.monster_ids
        ae() do j
            count(m -> m["id"] == monster, monsters(j))
        end
    end
    ae() do j
        minimum(m -> m["current_hp"], monsters(j))
    end
    ae() do j
        maximum(m -> m["current_hp"], monsters(j))
    end
    encoder
end

end # module
