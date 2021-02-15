using Flux
using OpenAIGym

struct VanillaNetwork{N}
    network :: N
end

Flux.@functor VanillaNetwork

function VanillaNetwork(in, out, hidden)
    layers = Any[Dense(in, hidden[1], mish, initW=Flux.kaiming_uniform)]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], mish, initW=Flux.kaiming_uniform))
    end
    push!(layers, Dense(hidden[end], out, identity))
    push!(layers, softmax)
    VanillaNetwork(Chain(layers...))
end

function (p::VanillaNetwork)(s)
    p.network(s)
end

struct DuellingNetwork{N, A, V}
    main_network   :: N
    action_network :: A
    value_network  :: V
    function DuellingNetwork(in, out, hidden)
        main_layers = Any[Dense(in, hidden[1], mish, initW=Flux.kaiming_uniform)]
        for i in 1:length(hidden)-1
            push!(main_layers, Dense(hidden[i], hidden[i+1], mish, initW=Flux.kaiming_uniform))
        end
        main_network = Chain(main_layers...)
        action_network = Dense(hidden[end], out, identity)
        value_network = Dense(hidden[end], 1, identity)
        new{typeof(main_network), typeof(action_network), typeof(value_network)}(main_network, action_network, value_network)
    end
end

Flux.@functor DuellingNetwork

function (qn::DuellingNetwork)(s)
    n = qn.main_network(s)
    a = qn.action_network(n)
    v = qn.value_network(n)
    a_mean = mean(a, dims=1)
    @assert typeof(a) == Array{Float32, 2}
    @assert typeof(v) == Array{Float32, 2}
    @assert typeof(a_mean) == Array{Float32, 2}
    v .+ a .- a_mean
end

function advantage(qn, s)
    q = qn(s)
    q_mean = mean(q, dims=1)
    @assert typeof(q) == Array{Float32, 2}
    @assert typeof(q_mean) == Array{Float32, 2}
    q .- q_mean
end

struct Policy <: AbstractPolicy
    a_to_i
    i_to_a
    policy_network
    q_network
end

function Reinforce.action(policy::Policy, r, s, A)
    a_p = policy.policy_network(s)
    sample(1:length(a_p), Weights(a_p)) |> policy.i_to_a
end

function mc_q(r, f, γ=1.0f0)
    result = similar(r)
    q = 0f0
    for i in length(r):-1:1
        q *= 1.0f0 - f[1, i]
        q += r[1, i]
        result[1, i] = q
        q *= γ
    end
    result
end

function onehot(hot_i, length)
    result = zeros(Float32, length)
    result[hot_i] = 1.0f0
    result
end

clip(n, ϵ) = clamp(n, 1 - ϵ, 1 + ϵ)

function optimize!(env, policy, sars, epochs=10_000, ϵ=0.2)
    stack(f) = Float32.(reduce(hcat, map(f, sars)))
    s = stack(x -> x[1])
    a = stack(x -> onehot(policy.a_to_i(x[2]), length(env.actions)))
    r = stack(x -> x[3])
    s′ = stack(x -> x[4])
    f = stack(x -> x[5])
    q = mc_q(r, f)
    @assert typeof(s) == Array{Float32, 2} typeof(s)
    @assert typeof(a) == Array{Float32, 2} typeof(a)
    @assert typeof(r) == Array{Float32, 2} typeof(r)
    @assert typeof(s′) == Array{Float32, 2} typeof(s′)
    @assert typeof(f) == Array{Float32, 2} typeof(f)
    @assert typeof(q) == Array{Float32, 2} typeof(q)
    q_opt = RMSProp()
    policy_opt = RMSProp()
    for epoch in 1:epochs
        i_sample = sample(1:size(s)[2], 100)
        s_sample = s[:, i_sample]
        a_sample = a[:, i_sample]
        q_sample = q[:, i_sample]
        grads = gradient(params(policy.q_network)) do
            predicted = sum(policy.q_network(s_sample) .* a_sample, dims=1)
            mean((predicted .- q_sample).^2)
        end
        Flux.Optimise.update!(q_opt, params(policy.q_network), grads)
    end
    target_policy_network = deepcopy(policy.policy_network)
    for epoch in 1:epochs
        i_sample = sample(1:size(s)[2], 100)
        s_sample = s[:, i_sample]
        a_sample = a[:, i_sample]
        advantage_sample = advantage(policy.q_network, s_sample) .* a_sample
        target_a_p = sum(target_policy_network(s_sample) .* a_sample, dims=1)
        grads = gradient(params(policy.policy_network)) do
            online_a_p = sum(policy.policy_network(s_sample) .* a_sample, dims=1)
            a_ratio = online_a_p ./ target_a_p
            a_ratio_clipped = clip.(a_ratio, ϵ)
            -mean(minimum([a_ratio .* advantage_sample; a_ratio_clipped .* advantage_sample], dims=1))
        end
        Flux.Optimise.update!(policy_opt, params(policy.policy_network), grads)
    end
end

function run_episodes(env, policy, n_episodes)
    sars = []
    rewards = []
    for episode in 1:n_episodes
        r = run_episode(env, policy) do (s, a, r, s′)
            push!(sars, (copy(s), a, r, copy(s′), finished(env)))
            if episode == 1; render(env) end
        end
        push!(rewards, r)
    end
    return sars, rewards
end

function run(generations=5000)
    env = GymEnv(:CartPole, :v1)
    rewards = []
    policy = Policy(
        x -> x+1,
        x -> x-1,
        VanillaNetwork(length(env.state), length(env.actions), [32]),
        DuellingNetwork(length(env.state), length(env.actions), [32]))
    try
        for gen in 1:generations
            sars, rs = run_episodes(env, policy, 100)
            append!(rewards, rs)
            println(mean(rs))
            if mean(rs) >= 475; break end
            optimize!(env, policy, sars)
        end
    catch e
        if !isa(e, InterruptException); rethrow() end
    finally
        close(env)
    end
    return policy, rewards
end
