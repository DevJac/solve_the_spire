module SARSM

export SARS, add_state, add_action, add_reward, awaiting, sar_state, sar_action, sar_reward

struct SARS
    states     :: Vector{Any}
    actions    :: Vector{Any}
    rewards    :: Vector{Tuple{Float32, Float32}}
end
SARS() = SARS([], [], [])

function add_state(sars::SARS, state)
    @assert all(v -> length(sars.states) == length(v), (sars.actions, sars.rewards))
    push!(sars.states, state)
end

function add_action(sars::SARS, action)
    @assert all(v -> length(sars.states)-1 == length(v), (sars.actions, sars.rewards))
    push!(sars.actions, action)
end

function add_reward(sars::SARS, reward, continuity=1.0f0)
    @assert all(v -> length(sars.rewards)+1 == length(v), (sars.states, sars.actions))
    push!(sars.rewards, (reward, continuity))
end

@enum SAR sar_state sar_action sar_reward

function awaiting(sars::SARS)
    if all(v -> length(sars.states) == length(v), (sars.actions, sars.rewards))
        return sar_state
    end
    if all(v -> length(sars.states)-1 == length(v), (sars.actions, sars.rewards))
        return sar_action
    end
    if all(v -> length(sars.rewards)+1 == length(v), (sars.states, sars.actions))
        return sar_reward
    end
end

end # module
