using Test
using Flux
using Utils
using Zygote

@testset "mc_q" begin
    falses = [false false false false]
    @test mc_q([1 2 3 4], falses) == [10 9 7 4]
    @test mc_q([1 1 1 1], falses) == [4 3 2 1]
    @test isapprox(mc_q([1 1 1 1], falses, 0.9), [3.439 2.71 1.9 1])
    @test mc_q([1 1 1 1], [false true false true]) == [2 1 2 1]
    @test mc_q([1 1 1 1], [false true false false]) == [2 1 2 1]
    @test isapprox(mc_q([1 1 1 1], [false true false true], 0.8), [1.8 1 1.8 1])
    @test isapprox(mc_q([1 1 1 1], [false true false false], 0.8), [1.8 1 1.8 1])
    @test isapprox(mc_q([1 1 1 1], [true false false false], 0.9), [1 2.71 1.9 1])
    @test isapprox(mc_q([1 1 1 1], [false false true false], 0.9), [2.71 1.9 1 1])
end

@testset "onehot" begin
    @test onehot(2, 4) == [0, 1, 0, 0]
    @test onehot(1, 5) == [1, 0, 0, 0, 0]
    @test onehot(3, 3) == [0, 0, 1]
end

@testset "clip" begin
    @test clip(2, 0.2) == 1.2
    @test clip(1, 0.2) == 1
    @test clip(0, 0.2) == 0.8
end

@testset "find" begin
    @test find(2, [1, 2, 3]) == 2
    @test find(4, [1, 2, 3]) == nothing
end

@testset "Smoother" begin
    s = Smoother(0.9)
    @test isapprox(smooth!(s, 1), 0.1)
    @test isapprox(smooth!(s, 1), 0.19)
    s = Smoother(0.1)
    @test isapprox(smooth!(s, 1), 0.9)
    @test isapprox(smooth!(s, 1), 0.99)
    s = Smoother(0.0)
    @test isapprox(smooth!(s, 1), 1)
    @test isapprox(smooth!(s, 2), 2)
    s = Smoother(0.1, initial_value=1)
    @test isapprox(smooth!(s, 2), 1.9)
    @test isapprox(smooth!(s, 2), 1.99)
    s = Smoother()
    @test isapprox(smooth!(s, 1), 0.1)
    @test isapprox(smooth!(s, 1), 0.19)
end

@testset "Batcher" begin
    data = collect(1:10)
    b = Batcher(data, 4)
    total = 0
    for (i, batch) in enumerate(b)
        @test length(batch) == 4
        total += sum(batch)
        if i >= 10; break end
    end
    @test total == sum(data) * 4
    b = Batcher(data, 20)
    batch, _ = iterate(b)
    @test length(batch) == 10
    @test sum(batch) == sum(data)
    b = Batcher(data, 10)
    batch, _ = iterate(b)
    @test length(batch) == 10
    @test sum(batch) == sum(data)
    batch, _ = iterate(b)
    @test length(batch) == 10
    @test sum(batch) == sum(data)
end

@testset "explore_odds" begin
    @test explore_odds([0.333, 0.333, 0.333]) == 0
    @test explore_odds([0.4, 0.3, 0.3]) == 0.6
    @test explore_odds([0.2, 0.2, 0.3, 0.3]) == 0.4
end

@testset "diagcat" begin
    x = [1 2; 3 4]
    y = reshape([5], 1, 1)
    z = [6 7 8; 6 7 8]
    @test diagcat(x, y, z) == [1 2 0 0 0 0
                               3 4 0 0 0 0
                               0 0 5 0 0 0
                               0 0 0 6 7 8
                               0 0 0 6 7 8]
    val, grad = valgrad(params(x, y, z)) do
        a = prod(x) + 2*sum(y) + sum(z.^2)
        b = diagcat(x, y, z)
        c = sum(b.^2)
        a + c
    end
    @test val == prod([1, 2, 3, 4]) + 2*5 + sum([6 7 8; 6 7 8].^2)*2 + 25 + 16 + 9 + 4 + 1
    @test grad[x] == [24 12; 8 6] .+ [2 4; 6 8]
    @test grad[y] == reshape([12], 1, 1)
    @test grad[z] == [24 28 32; 24 28 32]
end
