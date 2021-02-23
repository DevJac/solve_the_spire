using Test

using STSAgents

@testset "encode_card_in_hand" begin
    gd = GameData(["card_1", "card_2"], [], [], [])
    card_json = Dict("id" => "card_1", "upgrades" => 1, "cost" => 2)
    @test STSAgents.encode_card_in_hand(gd, card_json) == Float32.([1, 0, 0, 1, 1, 2])
    card_json["id"] = "card_2"
    @test STSAgents.encode_card_in_hand(gd, card_json) == Float32.([0, 1, 0, 1, 1, 2])
    card_json["id"] = "card_3"
    @test STSAgents.encode_card_in_hand(gd, card_json) == Float32.([0, 0, 1, 1, 1, 2])
    card_json["id"] = "card_4"
    @test STSAgents.encode_card_in_hand(gd, card_json) == Float32.([0, 0, 1, 1, 1, 2])
    card_json["cost"] = 3
    @test STSAgents.encode_card_in_hand(gd, card_json) == Float32.([0, 0, 1, 1, 1, 3])
    card_json["upgrades"] = 0
    @test STSAgents.encode_card_in_hand(gd, card_json) == Float32.([0, 0, 1, 1, 0, 3])
end

@testset "encode_cards_in_draw_discard_pile" begin
    gd = GameData(["card_1", "card_2"], [], [], [])
    cards_json = [
        Dict("id" => "card_1", "upgrades" => 1),
        Dict("id" => "card_1", "upgrades" => 0),
        Dict("id" => "card_2", "upgrades" => 0),
        Dict("id" => "card_3", "upgrades" => 1)
    ]
    @test STSAgents.encode_cards_in_draw_discard_pile(gd, cards_json) == Float32.([0.5, 0.5, 0.25, 0, 0.25, 1])
end

@testset "encode_potion" begin
    gd = GameData([], ["p1", "p2", "p3"], [], [])
    potion_json = Dict("id" => "p2")
    @test STSAgents.encode_potion(gd, potion_json) == Float32.([0, 1, 0])
end

@testset "encode_player_powers" begin
    gd = GameData([], [], ["pow_1", "pow_2"], [])
    player_powers_json = [
        Dict("id" => "pow_1", "amount" => -1),
        Dict("id" => "pow_3", "amount" => 2)
    ]
    @test STSAgents.encode_player_powers(gd, player_powers_json) == Float32.([1, -1, 0, 0, 1, 2])
end

@testset "encode_player" begin
    player_json = Dict(
        "current_hp" => 35,
        "max_hp" => 70,
        "block" => 6)
    @test STSAgents.encode_player(player_json) == Float32.([35, 70, 0.5, 6])
end
