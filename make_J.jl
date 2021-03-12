# To run: julia --project make_J.jl
using PackageCompiler
create_sysimage([
    :BSON
    :Flux
    :JSON
    :NNlib
    :OwnTime
    :StatsBase
    :TensorBoardLogger
    :Zygote
], sysimage_path="J")
