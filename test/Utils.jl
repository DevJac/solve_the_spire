using Test

using Utils

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

@testset "DiskStringSet" begin
    filename = "for_testing_DiskStringSet.delete_me.temp"
    s = DiskStringSet(filename)
    push!(s, "one")
    push!(s, "one")
    push!(s, "two")
    push!(s, "three")
    push!(s, "two")
    push!(s, "three")
    push!(s, " one")
    push!(s, "one ")
    push!(s, " two ")
    push!(s, "   three")
    push!(s, "   two       ")
    push!(s, "           three  ")
    @test length(s) == 3
    @test "two" in s
    pop!(s, "two")
    @test !("two" in s)
    @test length(s) == 2
    @test "one" in s
    @test "three" in s
    a = collect(s)
    @test length(a) == 2
    @test read(open(filename), String) == "one\nthree\n"
    rm(filename)
end