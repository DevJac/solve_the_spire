using JSON
using Test

using STSAgents

@testset "make_hand_card_encoder" begin
    gd = GameData(["card_1", "card_2"], [], [], [])
    encoder = STSAgents.make_hand_card_encoder(gd)
    # One-hot encoding for each card (2)
    # Card upgrades (1)
    # Card cost (1)
    @test length(encoder.encoders) == 2 + 1 + 1
    j = JSON.parse("""
        [

            {"id": "card_1", "upgrades": 1, "cost": 3},
            {"id": "card_1", "upgrades": 0, "cost": 0},
            {"id": "card_2", "upgrades": 0, "cost": 1}
        ]
    """)
    @test encode(encoder, j[1]) == Float32.([1, 0, 1, 3])
    @test encode(encoder, j[2]) == Float32.([1, 0, 0, 0])
    @test encode(encoder, j[3]) == Float32.([0, 1, 0, 1])
end

@testset "make_draw_discard_encoder" begin
    gd = GameData(["card_1", "card_2"], [], [], [])
    encoder = STSAgents.make_draw_discard_encoder(gd)
    # 2 encoded vectors for each card (2), in both piles
    @test length(encoder.encoders) == 2 * 2 * 2
    j = JSON.parse("""
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
    @test encode(encoder, j) == Float32.([0.5, 0, 0.5, 1, 2/3, 0.5, 1/3, 0])
end

@testset "make_player_encoder" begin
    gd = GameData([], [], ["power_1", "power_2"], [])
    encoder = STSAgents.make_player_encoder(gd)
    # 2 encoded vectors for each power (2)
    @test length(encoder.encoders) == 2 * 2 + 5
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
                        "block": 6
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
                        "block": 0
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
    @test encode(encoder, j1) == Float32.([1, -1, 1, 1, 30, 70, 30 / 70, 6, 6 - 15])
    @test encode(encoder, j2) == Float32.([0, 0, 1, 3, 20, 77, 20 / 77, 0, -25])
end
