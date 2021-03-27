using Flux
using Test

using Networks

@testset "value" begin
    q = QNetwork(4, 3, [])
    i = rand(4, 10)
    @test size(q(i)) == (3, 10)
    @test size(value(q, i)) == (1, 10)
end

@testset "advantage" begin
    p = PolicyNetwork(4, 3, [8, 9])
    result = advantage(p, rand(4, 10))
    @test size(result) == (3, 10)
    @test isapprox(sum(result, dims=1), zeros(1, 10), atol=1e-5)
end

@testset "PolicyNetwork" begin
    p = PolicyNetwork(4, 3, [8, 9])
    @test params(p) .|> length == [32, 8, 72, 9, 27, 3]
    result = p(rand(4, 10))
    @test size(result) == (3, 10)
    @test all(result) do n
        0 <= n && n <= 1
    end
    @test isapprox(sum(result, dims=1), ones(1, 10))
    hidden_layers = [8, 9]
    p = PolicyNetwork(4, 3, hidden_layers)
    @test params(p) .|> length == [32, 8, 72, 9, 27, 3]
    p = PolicyNetwork(4, 3, hidden_layers)
    @test params(p) .|> length == [32, 8, 72, 9, 27, 3]
end

@testset "QNetwork" begin
    q = QNetwork(4, 3, [8, 9])
    @test params(q) .|> length == [32, 8, 72, 9, 27, 3, 9, 1]
    @test size(q(rand(4, 10))) == (3, 10)
    hidden_layers = [8, 9]
    q = QNetwork(4, 3, hidden_layers)
    @test params(q) .|> length == [32, 8, 72, 9, 27, 3, 9, 1]
    @test size(q(rand(4, 10))) == (3, 10)
    q = QNetwork(4, 3, hidden_layers)
    @test params(q) .|> length == [32, 8, 72, 9, 27, 3, 9, 1]
    @test size(q(rand(4, 10))) == (3, 10)
end

@testset "VanillaNetwork" begin
    n = VanillaNetwork(4, 3, [8, 9])
    @test params(n) .|> length == [40, 8, 72, 9, 27, 3]
    @test size(n(rand(4, 10))) == (3, 10)
    @test length(n) == 3
    n = VanillaNetwork(4, 3, [8])
    @test params(n) .|> length == [40, 8, 24, 3]
    @test size(n(rand(4, 10))) == (3, 10)
    n = VanillaNetwork(4, 3, [])
    @test params(n) .|> length == [15, 3]
    @test size(n(rand(4, 10))) == (3, 10)
    hidden_layers = [8, 9]
    n = VanillaNetwork(4, 3, hidden_layers)
    @test params(n) .|> length == [40, 8, 72, 9, 27, 3]
    @test size(n(rand(4, 10))) == (3, 10)
    n = VanillaNetwork(4, 4, hidden_layers)
    @test params(n) .|> length == [40, 8, 72, 9, 36, 4]
    @test size(n(rand(4, 10))) == (4, 10)
    @test length(n) == 4
end

@testset "PoolNetwork" begin
    n = PoolNetwork(4, 3, [8, 9])
    @test params(n) .|> length == [40, 8, 72, 9, 27, 3, 12]
    @test size(n(rand(4, 10))) == (3,)
    @test length(n) == 3
    n = PoolNetwork(4, 3, [8])
    @test params(n) .|> length == [40, 8, 24, 3, 12]
    @test size(n(rand(4, 10))) == (3,)
    n = PoolNetwork(4, 3, [])
    @test params(n) .|> length == [15, 3, 12]
    @test size(n(rand(4, 10))) == (3,)
    hidden_layers = [8, 9]
    n = PoolNetwork(4, 3, hidden_layers)
    @test params(n) .|> length == [40, 8, 72, 9, 27, 3, 12]
    @test size(n(rand(4, 10))) == (3,)
    n = PoolNetwork(4, 4, hidden_layers)
    @test params(n) .|> length == [40, 8, 72, 9, 36, 4, 16]
    @test size(n(rand(4, 10))) == (4,)
    @test length(n) == 4
end

@testset "PoolEachNetwork" begin
    n = PoolEachNetwork(5, 3, [8, 9])
    @test length(n) == 4
    @test params(n) .|> length == [48, 8, 72, 9, 18, 2, 48, 8, 72, 9, 18, 2, 8]
    out = n(rand(5, 10))
    @test size(out) == (4, 10)
    @test !all(x -> out[1,1] == x, out[1,:])
    @test !all(x -> out[2,1] == x, out[2,:])
    @test all(x -> out[3,1] == x, out[3,:])
    @test all(x -> out[4,1] == x, out[4,:])
end
