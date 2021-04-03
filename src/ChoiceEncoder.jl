module Encoders
using Zygote

struct ChoiceEncoder
    state_embedders  :: Dict{Symbol, Any}
    choice_embedders :: Dict{Symbol, Any}
    pooler           :: Any
    states_encoded   :: Dict{Symbol, Any}
    choices_encoded  :: Dict{Symbol, Any}
    choice_actions   :: Dict{Symbol, Vector{Any}}
end
function ChoiceEncoder(state_embedders, choice_embedders, pooler)
    presence_indicators = length(ce.choice_embedders) > 1 ? length(ce.choice_embedders) : 0
    if sum(length, choice_embedder) + presence_indicators != length(pooler)
        throw("Choice embedders and pooler are different lengths")
    end
    ChoiceEncoder(
        state_embedders,
        choice_embedders,
        pooler,
        Dict(k => nothing for k in keys(state_embedders)),
        Dict(k => nothing for k in keys(choice_embedders)),
        Dict(k => [] for k in keys(choice_embedders)))
end

function Base.length(ce::ChoiceEncoder)
    state_embedders_total_length = sum(length, values(ce.state_embedders))
    choice_embedders_total_length = sum(length, values(ce.choice_embedders))
    pooler_length = length(ce.pooler)
    presence_indicators = length(ce.choice_embedders) > 1 ? length(ce.choice_embedders) : 0
    state_embedders_total_length + choice_embedders_total_length + presence_indicators
end

function add_state(ce::ChoiceEncoder, state_symbol, state_encoded)
    if length(state_encoded) != length(ce.states_embedders[state_symbol])
        throw("State and state embedder are different lengths")
    end
    ce.states_encoded[state_symbol] = state_encoded
end

function add_choice(ce::ChoiceEncoder, choice_symbol, choice_encoded, choice_action)
    if length(choice_encoded) != length(ce.choice_embedders[choice_symbol])
        throw("Choice and choice embedder are different lengths")
    end
    ce.choices_encoded[choice_symbol] = choice_encoded
    ce.choice_actions[choice_symbol] = choice_action
end

function encode_choices(ce::ChoiceEncoder)
    if any(isnothing, values(states_encoded))
        throw("Missing encoded states")
    end
    presence_indicators = length(ce.choice_embedders) > 1 ? length(ce.choice_embedders) : 0
    state_embedded = reduce(vcat, [ce.state_embedders[sym](ce.states_encoded[sym]) for sym in sort(keys(ce.states_encoded))])
    Zygote.@ignore @assert length(state_embedded) == sum(length, values(ce.state_embedders))
    choice_categories_embedded = [reduce(hcat, ce.choice_embedders[sym].(ce.choices_encoded[sym])) for sym in sort(keys(ce.choices_encoded))]
    if presence_indicators > 0
        choices_embedded = reduce(diagcat, map(onetop, choice_categories_embedded))
    else
        choices_embedded = reduce(diagcat, choice_categories_embedded)
    end
    choice_actions = reduce(vcat, [ce.choice_actions[sym] for sym in sort(keys(ce.choice_actions))])
    Zygote.@ignore @assert size(choices_embedded, 2) == length(choice_actions)
    pool = ce.pooler(choices_embedded)
    final_encoding = vcat(
        repeat(state_embedded, 1, size(choices_embedded, 2)),
        repeat(pool, 1, size(choices_embedded, 2)),
        choices_embedded)
    Zygote.@ignore @assert size(final_encoding, 2) == length(choice_actions)
    final_encoding, choice_actions
end

onetop(x) = [zeros(size(x, 2)); x]

end # module
