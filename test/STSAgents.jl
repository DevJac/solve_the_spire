using JSON
using Test

using STSAgents

@testset "make_hand_encoder" begin
    gd = GameData(["card_1", "card_2"], [], [], [])
    encoder = STSAgents.make_hand_encoder(gd)
    # One-hot encoding for each card (2)
    # One-hot encoding for an unknown card (1)
    # Card present (1)
    # Card upgrades (1)
    # Card cost (1)
    # Up to 10 cards in hand
    @test length(encoder.encoders) == 6 * 10
    j = JSON.parse("""
        {"game_state": {"combat_state": {"hand": [
            {"id": "card_1", "upgrades": 1, "cost": 3},
            {"id": "card_1", "upgrades": 0, "cost": 0},
            {"id": "card_2", "upgrades": 0, "cost": 1},
            {"id": "card_3", "upgrades": 2, "cost": 2}
        ]}}}
    """)
    @test encode(encoder, j) == Float32.([1, 0, 0, 1, 1, 3,
                                          1, 0, 0, 1, 0, 0,
                                          0, 1, 0, 1, 0, 1,
                                          0, 0, 1, 1, 2, 2,
                                          0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0])
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
