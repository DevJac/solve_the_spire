module ChoiceEncoders
using Flux
using Networks
using Utils
using Zygote

export ChoiceEncoder, add_encoded_state, add_encoded_choice, encode_choices, encode_state

struct ChoiceEncoder
    state_embedders  :: Dict{Symbol, Any}
    choice_embedders :: Dict{Symbol, Any}
    pooler           :: Any
    states_encoded   :: Dict{Symbol, Any}
    choices_encoded  :: Dict{Symbol, Vector{Any}}
    choice_actions   :: Dict{Symbol, Vector{Any}}
end

function ChoiceEncoder(state_embedders, choice_embedders, pool_out, pool_hidden)
    presence_indicators = length(choice_embedders) > 1 ? length(choice_embedders) : 0
    ChoiceEncoder(
        state_embedders,
        choice_embedders,
        PoolNetwork(sum(length, values(choice_embedders)) + presence_indicators, pool_out, pool_hidden),
        Dict(k => nothing for k in keys(state_embedders)),
        Dict(k => [] for k in keys(choice_embedders)),
        Dict(k => [] for k in keys(choice_embedders)))
end

function ChoiceEncoder(state_embedders, choice_embedders, pool)
    ChoiceEncoder(
        state_embedders,
        choice_embedders,
        pool,
        Dict(k => nothing for k in keys(state_embedders)),
        Dict(k => [] for k in keys(choice_embedders)),
        Dict(k => [] for k in keys(choice_embedders)))
end

function reset!(ce::ChoiceEncoder)
    ce.states_encoded = Dict(k => nothing for k in keys(ce.state_embedders))
    ce.choices_encoded = Dict(k => [] for k in keys(ce.choice_embedders))
    ce.choie_actions = Dict(k => [] for k in keys(ce.choice_embedders))
end

function Flux.trainable(ce::ChoiceEncoder)
    trainables = []
    append!(trainables, [Flux.trainable(n) for n in values(ce.state_embedders)])
    append!(trainables, [Flux.trainable(n) for n in values(ce.choice_embedders)])
    push!(trainables, Flux.trainable(ce.pooler))
    tuple(trainables...)
end

function Base.length(ce::ChoiceEncoder)
    state_embedders_total_length = sum(length, values(ce.state_embedders))
    choice_embedders_total_length = sum(length, values(ce.choice_embedders))
    pooler_length = length(ce.pooler)
    presence_indicators = length(ce.choice_embedders) > 1 ? length(ce.choice_embedders) : 0
    state_embedders_total_length + choice_embedders_total_length + presence_indicators
end

function add_encoded_state(ce::ChoiceEncoder, state_symbol, state_encoded)
    ce.states_encoded[state_symbol] = state_encoded
end

function add_encoded_choice(ce::ChoiceEncoder, choice_symbol, choice_encoded, choice_action)
    push!(ce.choices_encoded[choice_symbol], choice_encoded)
    push!(ce.choice_actions[choice_symbol], choice_action)
end

function encode_choices(ce::ChoiceEncoder)
    if any(isnothing, values(ce.states_encoded))
        throw("Missing encoded states")
    end
    state_keys = Zygote.@ignore sort(collect(keys(ce.state_embedders)))
    choice_keys = Zygote.@ignore sort(collect(keys(ce.choice_embedders)))
    presence_indicators = length(ce.choice_embedders) > 1 ? length(ce.choice_embedders) : 0
    state_embedded = reduce(
        vcat,
        [ce.state_embedders[sym](ce.states_encoded[sym]) for sym in state_keys])
    Zygote.@ignore @assert length(state_embedded) == sum(length, values(ce.state_embedders))
    choice_categories_embedded = [
        length(ce.choices_encoded[sym]) >= 1 ? reduce(hcat, ce.choice_embedders[sym].(ce.choices_encoded[sym])) : mask(ce, sym)
        for sym in choice_keys]
    if presence_indicators > 0
        choices_embedded = reduce(diagcat, map(onetop, choice_categories_embedded))
    else
        choices_embedded = reduce(diagcat, choice_categories_embedded)
    end
    choice_actions = reduce(vcat, [ce.choice_actions[sym] for sym in choice_keys])
    Zygote.@ignore @assert size(choices_embedded, 2) == length(choice_actions)
    choices_embedded = choices_embedded[:, findall(a -> !isa(a, MaskAction), choice_actions)]
    choice_actions = choice_actions[findall(a -> !isa(a, MaskAction), choice_actions)]
    pool = ce.pooler(choices_embedded)
    final_encoding = vcat(
        repeat(state_embedded, 1, size(choices_embedded, 2)),
        repeat(pool, 1, size(choices_embedded, 2)),
        choices_embedded)
    Zygote.@ignore @assert size(final_encoding, 2) == length(choice_actions)
    final_encoding, choice_actions
end

function encode_state(ce::ChoiceEncoder)
    if any(isnothing, values(ce.states_encoded))
        throw("Missing encoded states")
    end
    state_keys = Zygote.@ignore sort(collect(keys(ce.state_embedders)))
    choice_keys = Zygote.@ignore sort(collect(keys(ce.choice_embedders)))
    presence_indicators = length(ce.choice_embedders) > 1 ? length(ce.choice_embedders) : 0
    state_embedded = reduce(
        vcat,
        [ce.state_embedders[sym](ce.states_encoded[sym]) for sym in state_keys])
    Zygote.@ignore @assert length(state_embedded) == sum(length, values(ce.state_embedders))
    choice_categories_embedded = [
        length(ce.choices_encoded[sym]) >= 1 ? reduce(hcat, ce.choice_embedders[sym].(ce.choices_encoded[sym])) : mask(ce, sym)
        for sym in choice_keys]
    if presence_indicators > 0
        choices_embedded = reduce(diagcat, map(onetop, choice_categories_embedded))
    else
        choices_embedded = reduce(diagcat, choice_categories_embedded)
    end
    choice_actions = reduce(vcat, [ce.choice_actions[sym] for sym in choice_keys])
    Zygote.@ignore @assert size(choices_embedded, 2) == length(choice_actions)
    choices_embedded = choices_embedded[:, findall(a -> !isa(a, MaskAction), choice_actions)]
    choice_actions = choice_actions[findall(a -> !isa(a, MaskAction), choice_actions)]
    pool = ce.pooler(choices_embedded)
    vcat(state_embedded, pool)
end

struct MaskAction; end

function mask(ce::ChoiceEncoder, sym)
    push!(ce.choice_actions[sym], MaskAction())
    zeros(length(ce.choice_embedders[sym]))
end

onetop(x) = [ones(1, size(x, 2)); x]

end # module
