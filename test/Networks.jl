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
end

@testset "QNetwork" begin
    q = QNetwork(4, 3, [8, 9])
    @test params(q) .|> length == [32, 8, 72, 9, 27, 3, 9, 1]
    @test size(q(rand(4, 10))) == (3, 10)
end

@testset "VanillaNetwork" begin
    n = VanillaNetwork(4, 3, [8, 9])
    @test params(n) .|> length == [32, 8, 72, 9, 27, 3]
    @test size(n(rand(4, 10))) == (3, 10)
    n = VanillaNetwork(4, 3, [8])
    @test params(n) .|> length == [32, 8, 24, 3]
    @test size(n(rand(4, 10))) == (3, 10)
    n = VanillaNetwork(4, 3, [])
    @test params(n) .|> length == [12, 3]
    @test size(n(rand(4, 10))) == (3, 10)
end
