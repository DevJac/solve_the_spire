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

    sar_structs = fill_q(sars)
    @test map(x -> x.q, sar_structs) == Float32.([2, 1, 1, 7, 13, 12, 11, 10])
    empty!(sars)
    @test map(x -> x.q, fill_q(sars)) == Float32.([])
    @test length(sars.states) == 0
    @test length(sars.actions) == 0
    @test length(sars.rewards) == 0
end
