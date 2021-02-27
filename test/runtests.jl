using Test

@testset "Utils" begin
    include("Utils.jl")
end

@testset "Encoders" begin
    include("Encoders.jl")
end

@testset "Networks" begin
    include("Networks.jl")
end
