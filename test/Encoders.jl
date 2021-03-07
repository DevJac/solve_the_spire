using JSON
using Test

using Encoders

@testset "make_hand_card_encoder" begin
    gd = GameData(["card_1", "card_2"], [], [], [])
    encoder = make_hand_card_encoder(gd)
    # One-hot encoding for each card (2)
    # Card upgrades (1)
    # Card cost (1)
    @test length(encoder) == 2 + 1 + 1
    j = JSON.parse("""
        [

            {"id": "card_1", "upgrades": 1, "cost": 3},
            {"id": "card_1", "upgrades": 0, "cost": 0},
            {"id": "card_2", "upgrades": 0, "cost": 1}
        ]
    """)
    @test encoder(j[1]) == Float32.([1, 0, 1, 3])
    @test encoder(j[2]) == Float32.([1, 0, 0, 0])
    @test encoder(j[3]) == Float32.([0, 1, 0, 1])
end

@testset "make_draw_discard_encoder" begin
    gd = GameData(["card_1", "card_2"], [], [], [])
    encoder = make_draw_discard_encoder(gd)
    # 2 encoded vectors for each card (2), in both piles
    @test length(encoder) == 2 * 2 * 2
    j1 = JSON.parse("""
        {"game_state": {"combat_state": {
            "draw_pile": [
                {"id": "card_1", "upgrades": 0},
                {"id": "card_1", "upgrades": 0},
                {"id": "card_2", "upgrades": 1},
                {"id": "card_2", "upgrades": 1}
            ],
            "discard_pile": [
                {"id": "card_2", "upgrades": 0},
                {"id": "card_1", "upgrades": 1},
                {"id": "card_1", "upgrades": 0}
        ]}}}
    """)
    j2 = JSON.parse("""
        {"game_state": {"combat_state": {
            "draw_pile": [],
            "discard_pile": [
                {"id": "card_2", "upgrades": 0},
                {"id": "card_1", "upgrades": 2},
                {"id": "card_1", "upgrades": 0}
        ]}}}
    """)
    @test encoder(j1) == Float32.([0.5, 0, 0.5, 1, 2/3, 0.5, 1/3, 0])
    @test encoder(j2) == Float32.([0, 0, 0, 0, 2/3, 1, 1/3, 0])
end

@testset "make_player_encoder" begin
    gd = GameData([], [], ["power_1", "power_2"], [])
    encoder = make_player_encoder(gd)
    # 2 encoded vectors for each power (2)
    # current_health, max_health, health_ratio, energy, block, surplus block (6)
    @test length(encoder) == 2 * 2 + 6
    j1 = JSON.parse("""
        {
            "game_state": {
                "combat_state": {
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
    @test encoder(j1) == Float32.([1, -1, 1, 1, 30, 70, 30 / 70, 3, 6, 6 - 15])
    @test encoder(j2) == Float32.([0, 0, 1, 3, 20, 77, 20 / 77, 0, 0, -25])
end
