module SARSM
using Statistics

export SARS, add_state, add_action, add_reward, awaiting, sar_state, sar_action, sar_reward

struct SARS
    states     :: Vector{Any}
    actions    :: Vector{Any}
    rewards    :: Vector{Tuple{Float32, Float32}}
end
SARS() = SARS([], [], [])

function Base.empty!(sars::SARS)
    empty!(sars.states)
    empty!(sars.actions)
    empty!(sars.rewards)
end

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

@enum SARPart sar_state sar_action sar_reward

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

export SingleSAR, fill_q

struct SingleSAR{S, A}
    state      :: S
    action     :: A
    reward     :: Float32
    continuity :: Float32
    q          :: Float32
    q_norm     :: Float32
end

function fill_q(sars::SARS, discount_factor=1.0f0)
    @assert all(v -> length(sars.states) == length(v), (sars.actions, sars.rewards))
    first_pass = SingleSAR[]
    q = 0.0f0
    for i in length(sars.rewards):-1:1
        reward, continuity = sars.rewards[i]
        q *= continuity * discount_factor
        q += reward
        push!(first_pass, SingleSAR(sars.states[i], sars.actions[i], reward, continuity, q, 0f0))
    end
    q_mean = mean([sar.q for sar in first_pass])
    q_std = std([sar.q for sar in first_pass])
    reverse([
        SingleSAR(
            sar.state,
            sar.action,
            sar.reward,
            sar.continuity,
            sar.q,
            ((sar.q - q_mean) / q_std))
        for sar in first_pass])
end

end # module
