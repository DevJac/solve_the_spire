module SARSM

export SARS, add_state, add_action, add_reward

struct SARS
    states  :: Vector{Any}
    actions :: Vector{Any}
    rewards :: Vector{Float32}
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

function add_reward(sars::SARS, reward)
    @assert all(v -> length(sars.rewards)+1 == length(v), (sars.states, sars.actions))
    push!(sars.rewards, reward)
end

end # module
