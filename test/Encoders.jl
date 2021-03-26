using JSON
using Test

using Encoders

@testset "make_card_encoder" begin
    gd = GameData([], ["card_1", "card_2"], ["COMMON", "RARE"], ["ATTACK", "DEFEND"], [], [], [], [], [])
    encoder = make_card_encoder(gd)
    # 1 to indicate presence of card (1)
    # One-hot encoding for each card (2)
    # One-hot encoding for each rarity (2)
    # One-hot encoding for each type (2)
    # Has targets (1)
    # Card cost (1)
    # Card upgrades (1)
    @test length(encoder) == 1 + 2 + 2 + 2 + 1 + 1 + 1
    j = JSON.parse("""
        [
            {"id": "card_1", "upgrades": 1, "cost": 3, "rarity": "RARE", "type": "ATTACK", "has_target": true},
            {"id": "card_1", "upgrades": 0, "cost": 0, "rarity": "COMMON", "type": "ATTACK", "has_target": false},
            {"id": "card_2", "upgrades": 0, "cost": 1, "rarity": "COMMON", "type": "DEFEND", "has_target": false}
        ]
    """)
    @test encoder(j[1]) == Float32.([1, 1, 0, 0, 1, 1, 0, 1, 3, 1])
    @test encoder(j[2]) == Float32.([1, 1, 0, 1, 0, 1, 0, 0, 0, 0])
    @test encoder(j[3]) == Float32.([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
end

@testset "player_basic_encoder" begin
    # Current HP, max HP, HP ratio, gold (4)
    @test length(player_basic_encoder) == 4
    j = JSON.parse("""
        {"game_state": {
            "current_hp": 20,
            "max_hp": 40,
            "gold": 99
        }}
    """)
    @test player_basic_encoder(j) == Float32.([20, 40, 0.5, 99])
end

@testset "make_player_combat_encoder" begin
    gd = GameData([], [], [], [], [], [], ["power_1", "power_2"], [], [])
    encoder = make_player_combat_encoder(gd)
    # 2 encoded vectors for each power (2*2)
    # Current HP, max HP, HP ratio, energy, block, surplus block, cards discarded this turn (7)
    @test length(encoder) == 2*2 + 7
    j1 = JSON.parse("""
        {
            "game_state": {
                "combat_state": {
                    "cards_discarded_this_turn": 2,
                    "player": {
                        "powers": [
                            {"id": "power_1", "amount": -1},
                            {"id": "power_2", "amount": 1}
                        ],
                        "current_hp": 30,
                        "max_hp": 70,
                        "block": 6,
                        "energy": 3
                    },
                    "monsters": [
                        {"intent": "ATTACK", "move_hits": 1, "move_adjusted_damage": 5},
                        {"intent": "ATTACK", "move_hits": 1, "move_adjusted_damage": 5},
                        {"intent": "ATTACK", "move_hits": 1, "move_adjusted_damage": 5}
                    ]
                }
            }
        }
    """)
    j2 = JSON.parse("""
        {
            "game_state": {
                "combat_state": {
                    "cards_discarded_this_turn": 0,
                    "player": {
                        "powers": [
                            {"id": "power_2", "amount": 3}
                        ],
                        "current_hp": 20,
                        "max_hp": 77,
                        "block": 0,
                        "energy": 0
                    },
                    "monsters": [
                        {"intent": "ATTACK", "move_hits": 1, "move_adjusted_damage": 5},
                        {"intent": "ATTACK", "move_hits": 1, "move_adjusted_damage": 10},
                        {"intent": "ATTACK", "move_hits": 2, "move_adjusted_damage": 5}
                    ]
                }
            }
        }
    """)
    @test encoder(j1) == Float32.([1, -1, 1, 1, 30, 70, 30 / 70, 3, 2, 6, 6 - 15])
    @test encoder(j2) == Float32.([0, 0, 1, 3, 20, 77, 20 / 77, 0, 0, 0, -25])
end

@testset "make_monster_encoder" begin
    gd = GameData([], [], [], [], ["monster_1", "monster_2"], ["m_power_1", "m_power_2"], [], [], [])
    encoder = make_monster_encoder(gd)
    # 1 to indicate presence of monster (1)
    # One-hot encoded monster (2)
    # 2 encoded vectors for each power (2*2)
    # Current HP, max HP, HP ratio, block, move damage, move hits, total damage (7)
    @test length(encoder) == 1 + 2 + 2*2 + 7
    j1 = JSON.parse("""
        {
                "powers": [
                    {"amount": 2, "id": "m_power_1"}
                ],
                "move_hits": 4,
                "intent": "ATTACK",
                "id": "monster_1",
                "block": 9,
                "current_hp": 4,
                "max_hp": 8,
                "move_adjusted_damage": 3
        }
    """)
    @test encoder(j1) == Float32.([1, 1, 0, 1, 2, 0, 0, 4, 8, 1/2, 9, 3, 4, 3*4])
end

@testset "make_relics_encoder" begin
    gd = GameData([], [], [], [], [], [], [], [], ["relic_1", "relic_2", "relic_3"])
    encoder = make_relics_encoder(gd)
    # 2 encoded vectors for each relic (2*3)
    @test length(encoder) == 2*3
    j = JSON.parse("""
        [
            {"id": "relic_1", "counter": -1},
            {"id": "relic_3", "counter": 2}
        ]
    """)
    @test encoder(j) == Float32.([1, -1, 0, 0, 1, 2])
end

@testset "make_potions_encoder" begin
    gd = GameData([], [], [], [], [], [], [], ["potion_1", "potion_2", "potion_3"], [])
    encoder = make_potions_encoder(gd)
    # Count encoding for each potion (3)
    @test length(encoder) == 3
    j = JSON.parse("""
        [
            {"id": "potion_1"},
            {"id": "potion_3"},
            {"id": "potion_3"}
        ]
    """)
    @test encoder(j) == Float32.([1, 0, 2])
end

@testset "map_encoder" begin
    # One-hot encoded next room type (6)
    # Min/max encoded next two rooms (2*6)
    # Min/max encoded remaining rooms (2*6)
    # One-hot encoded act boss (5 currently)
    # Floor (1)
    @test length(map_encoder) == 6*5 + 5 + 1
    j = JSON.parse(read("test/map.json", String))
    @test map_encoder(j, 0, 0) == Float32.([
        0, 0, 0, 1, 1,
        0, 0, 0, 1, 1,
        0, 0, 0, 3, 3,
        1, 2, 2, 4, 7,
        0, 0, 0, 0, 1,
        0, 0, 0, 3, 5,
        0, 0, 0, 1, 0, 3])
end
