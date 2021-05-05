using Flux
using Test

using Networks

@testset "VanillaNetwork" begin
    n = VanillaNetwork(4, 3, [8, 9])
    @test params(n) .|> length == [32, 8, 72, 9, 27, 3]
    @test size(n(rand(4, 10))) == (3, 10)
    @test length(n) == 3
    n = VanillaNetwork(4, 3, [8])
    @test params(n) .|> length == [32, 8, 24, 3]
    @test size(n(rand(4, 10))) == (3, 10)
    n = VanillaNetwork(4, 3, [])
    @test params(n) .|> length == [12, 3]
    @test size(n(rand(4, 10))) == (3, 10)
    hidden_layers = [8, 9]
    n = VanillaNetwork(4, 3, hidden_layers)
    @test params(n) .|> length == [32, 8, 72, 9, 27, 3]
    @test size(n(rand(4, 10))) == (3, 10)
    n = VanillaNetwork(4, 4, hidden_layers)
    @test params(n) .|> length == [32, 8, 72, 9, 36, 4]
    @test size(n(rand(4, 10))) == (4, 10)
    @test length(n) == 4
end

@testset "PoolNetwork" begin
    n = PoolNetwork(4, 3, [8, 9])
    @test params(n) .|> length == [32, 8, 72, 9, 27, 3, 36]
    @test size(n(rand(4, 10))) == (3,)
    @test length(n) == 3
    n = PoolNetwork(4, 3, [8])
    @test params(n) .|> length == [32, 8, 24, 3, 36]
    @test size(n(rand(4, 10))) == (3,)
    n = PoolNetwork(4, 3, [])
    @test params(n) .|> length == [12, 3, 36]
    @test size(n(rand(4, 10))) == (3,)
    hidden_layers = [8, 9]
    n = PoolNetwork(4, 3, hidden_layers)
    @test params(n) .|> length == [32, 8, 72, 9, 27, 3, 36]
    @test size(n(rand(4, 10))) == (3,)
    n = PoolNetwork(4, 4, hidden_layers)
    @test params(n) .|> length == [32, 8, 72, 9, 36, 4, 48]
    @test size(n(rand(4, 10))) == (4,)
    @test length(n) == 4
end
