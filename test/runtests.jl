using Test

@testset "Utils" begin
    include("Utils.jl")
end

@testset "Networks" begin
    include("Networks.jl")
end

@testset "STSAgents" begin
    include("STSAgents.jl")
end
