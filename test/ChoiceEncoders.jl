using Test
using Flux
using Networks
using Zygote

using ChoiceEncoders

struct MockNetwork
    in
    out
    out_value
    ispool
    MockNetwork(in, out, out_value, ispool=false) = new(in, out, out_value, ispool)
end

(n::MockNetwork)(s) = fill(n.out_value, n.out, n.ispool ? 1 : size(s, 2))

Base.length(n::MockNetwork) = n.out

@testset "happy_path" begin
    ce = ChoiceEncoder(
        Dict(:state_a => MockNetwork(3, 2, 1), :state_b => MockNetwork(4, 3, 2)),
        Dict(:choice_a => MockNetwork(5, 4, 3), :choice_b => MockNetwork(6, 5, 4)),
        MockNetwork(11, 7, 5, true))
    add_encoded_state(ce, :state_a, rand(3))
    add_encoded_state(ce, :state_b, rand(4))
    add_encoded_choice(ce, :choice_a, rand(5), 1)
    add_encoded_choice(ce, :choice_a, rand(5), 2)
    add_encoded_choice(ce, :choice_b, rand(6), 3)
    r = encode_choices(ce)
    @test r[2] == [1, 2, 3]
    @test r[1] == Float32.([1 1 1
                            1 1 1
                            2 2 2
                            2 2 2
                            2 2 2
                            5 5 5
                            5 5 5
                            5 5 5
                            5 5 5
                            5 5 5
                            5 5 5
                            5 5 5
                            1 1 0
                            3 3 0
                            3 3 0
                            3 3 0
                            3 3 0
                            0 0 1
                            0 0 4
                            0 0 4
                            0 0 4
                            0 0 4
                            0 0 4])
    ChoiceEncoders.reset!(ce)
    add_encoded_state(ce, :state_a, rand(3))
    add_encoded_state(ce, :state_b, rand(4))
    add_encoded_choice(ce, :choice_a, rand(5), 1)
    add_encoded_choice(ce, :choice_a, rand(5), 2)
    add_encoded_choice(ce, :choice_b, rand(6), 3)
    r = encode_choices(ce)
    @test r[2] == [1, 2, 3]
    @test r[1] == Float32.([1 1 1
                            1 1 1
                            2 2 2
                            2 2 2
                            2 2 2
                            5 5 5
                            5 5 5
                            5 5 5
                            5 5 5
                            5 5 5
                            5 5 5
                            5 5 5
                            1 1 0
                            3 3 0
                            3 3 0
                            3 3 0
                            3 3 0
                            0 0 1
                            0 0 4
                            0 0 4
                            0 0 4
                            0 0 4
                            0 0 4])
end

@testset "happy_path_with_real_networks" begin
    ce = ChoiceEncoder(
        Dict(:state_a => VanillaNetwork(3, 2, [1]), :state_b => VanillaNetwork(4, 3, [2])),
        Dict(:choice_a => VanillaNetwork(5, 4, [3]), :choice_b => NullNetwork()),
        7, [5])
    add_encoded_state(ce, :state_a, rand(3))
    add_encoded_state(ce, :state_b, rand(4))
    add_encoded_choice(ce, :choice_a, rand(5), 1)
    add_encoded_choice(ce, :choice_a, rand(5), 2)
    add_encoded_choice(ce, :choice_b, nothing, 3)
    r = encode_choices(ce)
    @test length(ce) == 18
    @test size(r[1]) == (18, 3)
    @test r[2] == [1, 2, 3]
end

@testset "no_choices_of_one_category" begin
    ce = ChoiceEncoder(
        Dict(:state_a => MockNetwork(3, 2, 1), :state_b => MockNetwork(4, 3, 2)),
        Dict(:choice_a => MockNetwork(5, 4, 3), :choice_b => MockNetwork(6, 5, 4)),
        MockNetwork(11, 7, 5, true))
    add_encoded_state(ce, :state_a, rand(3))
    add_encoded_state(ce, :state_b, rand(4))
    add_encoded_choice(ce, :choice_a, rand(5), 1)
    add_encoded_choice(ce, :choice_a, rand(5), 2)
    r = encode_choices(ce)
    @test r[2] == [1, 2]
    @test r[1] == Float32.([1 1
                            1 1
                            2 2
                            2 2
                            2 2
                            5 5
                            5 5
                            5 5
                            5 5
                            5 5
                            5 5
                            5 5
                            1 1
                            3 3
                            3 3
                            3 3
                            3 3
                            0 0
                            0 0
                            0 0
                            0 0
                            0 0
                            0 0])
    ChoiceEncoders.reset!(ce)
    add_encoded_state(ce, :state_a, rand(3))
    add_encoded_state(ce, :state_b, rand(4))
    add_encoded_choice(ce, :choice_b, rand(6), 3)
    r = encode_choices(ce)
    @test r[2] == [3]
    @test r[1] == reshape(Float32.([1
                                    1
                                    2
                                    2
                                    2
                                    5
                                    5
                                    5
                                    5
                                    5
                                    5
                                    5
                                    0
                                    0
                                    0
                                    0
                                    0
                                    1
                                    4
                                    4
                                    4
                                    4
                                    4]), 23, 1)
end

@testset "params" begin
    n = VanillaNetwork(4, 3, [8])
    ce = ChoiceEncoder(
        Dict(:state_a => VanillaNetwork(4, 3, [8]), :state_b => VanillaNetwork(4, 3, [8])),
        Dict(:choice_a => VanillaNetwork(4, 3, [8]), :choice_b => NullNetwork()),
        3, [8])
    @test length(params(ce)) == 16
    @test length.(params(ce)) == [32, 8, 24, 3, 32, 8, 24, 3, 32, 8, 24, 3, 40, 8, 24, 3]
end

@testset "encode_state" begin
    ce = ChoiceEncoder(
        Dict(:state_a => MockNetwork(3, 2, 1), :state_b => MockNetwork(4, 3, 2)),
        Dict(:choice_a => MockNetwork(5, 4, 3), :choice_b => MockNetwork(6, 5, 4)),
        MockNetwork(11, 7, 5, true))
    add_encoded_state(ce, :state_a, rand(3))
    add_encoded_state(ce, :state_b, rand(4))
    add_encoded_choice(ce, :choice_a, rand(5), 1)
    add_encoded_choice(ce, :choice_a, rand(5), 2)
    add_encoded_choice(ce, :choice_b, rand(6), 3)
    r = encode_state(ce)
    @test state_length(ce) == 12 == length(r)
    @test r == reshape(Float32.([1
                                 1
                                 2
                                 2
                                 2
                                 5
                                 5
                                 5
                                 5
                                 5
                                 5
                                 5]), 12, 1)
end

@testset "null_networks" begin
    ce = ChoiceEncoder(
        Dict(:state_a => MockNetwork(3, 2, 1), :state_b => MockNetwork(4, 3, 2)),
        Dict(:choice_a => MockNetwork(5, 4, 3), :choice_b => NullNetwork()),
        MockNetwork(11, 7, 5, true))
    add_encoded_state(ce, :state_a, rand(3))
    add_encoded_state(ce, :state_b, rand(4))
    add_encoded_choice(ce, :choice_a, rand(5), 1)
    add_encoded_choice(ce, :choice_a, rand(5), 2)
    add_encoded_choice(ce, :choice_b, rand(6), 3)
    r = encode_choices(ce)
    @test r[2] == [1, 2, 3]
    @test r[1] == Float32.([1 1 1
                            1 1 1
                            2 2 2
                            2 2 2
                            2 2 2
                            5 5 5
                            5 5 5
                            5 5 5
                            5 5 5
                            5 5 5
                            5 5 5
                            5 5 5
                            1 1 0
                            3 3 0
                            3 3 0
                            3 3 0
                            3 3 0
                            0 0 1])
end

@testset "encode_choices_then_encode_state" begin
    ce = ChoiceEncoder(
        Dict(:state_a => VanillaNetwork(3, 2, [1]), :state_b => VanillaNetwork(4, 3, [2])),
        Dict(:choice_a => VanillaNetwork(5, 4, [3]), :choice_b => NullNetwork()),
        7, [5])
    add_encoded_state(ce, :state_a, rand(3))
    add_encoded_state(ce, :state_b, rand(4))
    add_encoded_choice(ce, :choice_a, rand(5), 1)
    add_encoded_choice(ce, :choice_a, rand(5), 2)
    add_encoded_choice(ce, :choice_b, nothing, 3)
    e, a = encode_choices(ce)
    @test a == [1, 2, 3]
    encode_state(ce)
    ChoiceEncoders.reset!(ce)
    add_encoded_state(ce, :state_a, rand(3))
    add_encoded_state(ce, :state_b, rand(4))
    add_encoded_choice(ce, :choice_b, nothing, 1)
    add_encoded_choice(ce, :choice_b, nothing, 2)
    e, a = encode_choices(ce)
    @test a == [1, 2]
    encode_state(ce)
end

@testset "gradient" begin
    ce = ChoiceEncoder(
        Dict(:state_a => VanillaNetwork(3, 2, [1]), :state_b => VanillaNetwork(4, 3, [2])),
        Dict(:choice_a => VanillaNetwork(5, 4, [3]), :choice_b => NullNetwork()),
        7, [5])
    add_encoded_state(ce, :state_a, rand(3))
    add_encoded_state(ce, :state_b, rand(4))
    add_encoded_choice(ce, :choice_a, rand(5), 1)
    add_encoded_choice(ce, :choice_a, rand(5), 2)
    add_encoded_choice(ce, :choice_b, nothing, 3)
    gradient(params(ce)) do
        e, a = encode_choices(ce)
        Zygote.@ignore @test a == [1, 2, 3]
        sum(e)
    end
    gradient(params(ce)) do
        s = encode_state(ce)
        sum(s)
    end
    ChoiceEncoders.reset!(ce)
    add_encoded_state(ce, :state_a, rand(3))
    add_encoded_state(ce, :state_b, rand(4))
    add_encoded_choice(ce, :choice_b, nothing, 1)
    add_encoded_choice(ce, :choice_b, nothing, 2)
    gradient(params(ce)) do
        e, a = encode_choices(ce)
        Zygote.@ignore @test a == [1, 2]
        sum(e)
    end
    gradient(params(ce)) do
        s = encode_state(ce)
        sum(s)
    end
end
