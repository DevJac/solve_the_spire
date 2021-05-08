using Statistics
using Test

using SARSM

@testset "fill_q" begin
    sars = SARS()

    add_state(sars, 1)
    add_action(sars, 1)
    add_reward(sars, 1)

    add_state(sars, 1)
    add_action(sars, 1)
    add_reward(sars, 1, 0)

    @test awaiting(sars) == sar_state
    add_state(sars, 1)
    @test awaiting(sars) == sar_action
    add_action(sars, 1)
    @test awaiting(sars) == sar_reward
    add_reward(sars, 1, 0)

    add_state(sars, 1)
    add_action(sars, 1)
    add_reward(sars, 7, 0)

    add_state(sars, 1)
    add_action(sars, 1)
    add_reward(sars, 1)

    add_state(sars, 1)
    add_action(sars, 1)
    add_reward(sars, 1)

    add_state(sars, 1)
    add_action(sars, 1)
    add_reward(sars, 1)

    add_state(sars, 1)
    add_action(sars, 1)
    add_reward(sars, 10)

    sar_structs = fill_q(sars, sar -> sar.q)
    @test map(x -> x.q, sar_structs) == Float32.([2, 1, 1, 7, 13, 12, 11, 10])
    @test map(x -> x.continuity, sar_structs) == Float32.([1, 0, 0, 0, 1, 1, 1, 1])
    @test map(x -> x.weight, sar_structs) == Float32.([1/2, 1/2, 1, 1, 1/4, 1/4, 1/4, 1/4])
    @test isapprox(sar_structs[1].q_norm, -1.00268)
    @test isapprox(sar_structs[1].advantage_norm, -1.00268)

    @test mean([x.q_norm for x in sar_structs]) == 0
    @test std([x.q_norm for x in sar_structs]) == 1
    @test mean([x.advantage_norm for x in sar_structs]) == 0
    @test std([x.advantage_norm for x in sar_structs]) == 1

    sar_structs = fill_q(sars, sar -> sar.q + 1)
    @test map(x -> x.q, sar_structs) == Float32.([2, 1, 1, 7, 13, 12, 11, 10])
    @test map(x -> x.advantage, sar_structs) == Float32.([3, 2, 2, 8, 14, 13, 12, 11])
    @test isapprox(sar_structs[1].q_norm, -1.00268)
    @test isapprox(sar_structs[1].advantage_norm, -1.00268)
    sar_structs = fill_q(sars, sar -> sar.q * 2)
    @test isapprox(sar_structs[1].q_norm, -1.00268)
    @test isapprox(sar_structs[1].advantage_norm, -1.00268)
    sar_structs = fill_q(sars, sar -> sar.q * 2)
    @test isapprox(sar_structs[1].q_norm, -1.00268)
    @test isapprox(sar_structs[1].advantage_norm, -1.00268)
    sar_structs = fill_q(sars, sar -> sar.q + randn())
    @test isapprox(sar_structs[1].q_norm, -1.00268)
    @test !isapprox(sar_structs[1].advantage_norm, -1.00268)

    empty!(sars)
    @test map(x -> x.q, fill_q(sars, sar -> sar.q)) == Float32.([])
    @test length(sars.states) == 0
    @test length(sars.actions) == 0
    @test length(sars.rewards) == 0
end
