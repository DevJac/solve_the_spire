module Encoders
using Zygote

struct ChoiceEncoder
    state_embedders  :: Dict{Symbol, Any}
    choice_embedders :: Dict{Symbol, Any}
    states_encoded  :: Dict{Symbol, Any}
    choices_encoded :: Dict{Symbol, Any}
    choice_actions   :: Dict{Symbol, Vector{Any}}
end
function ChoiceEncoder(state_embedders, choice_embedders)
    ChoiceEncoder(
        state_embedders,
        choice_embedders,
        Dict(k => nothing for k in keys(state_embedders)),
        Dict(k => nothing for k in keys(choice_embedders)))
end

function Base.length(ce::ChoiceEncoder)
    state_embedders_total_length = sum(length, values(ce.state_embedders))
    choice_embedders_total_length = sum(length, values(ce.choice_embedders))
    presence_indicators = length(ce.choice_embedders) > 1 ? length(ce.choice_embedders) : 0
    state_embedders_total_length + choice_embedders_total_length + presence_indicators
end

function add_state(ce::ChoiceEncoder, state_symbol, state_encoded)
    ce.states_encoded[state_symbol] = state_encoded
end

function add_choice(ce::ChoiceEncoder, choice_symbol, choice_encoded, choice_action)
    ce.choices_encoded[choice_symbol] = choice_encoded
    ce.choice_actions[choice_symbol] = choice_action
end

function encode_choices(ce::ChoiceEncoder)
    if any(isnothing, values(states_encoded))
        throw("Missing encoded states")
    end
    state_symbols = sort(keys(states_encoded))
    state_embedded = reduce(hcat, [ce.state_embedders[sym](ce.states_encoded[sym]) for sym in sort(keys(states_encoded))])
    # TODO: Assemble all choices using diagcat
    # TODO: Pool choices, ..., return encoded choices along with their respective actions
end

end # module
