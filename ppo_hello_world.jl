using Flux
using Networks
using OpenAIGym
using Utils

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

function optimize!(env, policy, sars, epochs=10_000, ϵ=0.2)
    stack(f) = Float32.(reduce(hcat, map(f, sars)))
    s = stack(x -> x[1])
    a = stack(x -> onehot(policy.a_to_i(x[2]), length(env.actions)))
    r = stack(x -> x[3])
    s′ = stack(x -> x[4])
    f = Bool.(stack(x -> x[5]))
    q = mc_q(r, f)
    @assert typeof(s) == Array{Float32, 2} typeof(s)
    @assert typeof(a) == Array{Float32, 2} typeof(a)
    @assert typeof(r) == Array{Float32, 2} typeof(r)
    @assert typeof(s′) == Array{Float32, 2} typeof(s′)
    @assert typeof(f) == BitArray{2} typeof(f)
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
        q_sample = q[:, i_sample]
        value_sample = mean(policy.q_network(s_sample), dims=1)
        advantage_sample = q_sample .- value_sample
        target_p = target_policy_network(s_sample)
        target_a_p = sum(target_p .* a_sample, dims=1)
        local online_p
        grads = gradient(params(policy.policy_network)) do
            online_p = policy.policy_network(s_sample)
            online_a_p = sum(online_p .* a_sample, dims=1)
            a_ratio = online_a_p ./ target_a_p
            a_ratio_clipped = clip.(a_ratio, ϵ)
            -mean(minimum([a_ratio .* advantage_sample; a_ratio_clipped .* advantage_sample], dims=1))
        end
        @assert !any(isnan, online_p)
        @assert !any(isnan, target_p)
        kl_div = Flux.Losses.kldivergence(online_p, target_p)
        @assert kl_div > -1e-6
        if kl_div >= 0.01
            println("        early stop on epoch: $epoch")
            break
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

function run(generations=50)
    env = GymEnv(:CartPole, :v1)
    rewards = []
    gens = 0
    policy = Policy(
        x -> x+1,
        x -> x-1,
        PolicyNetwork(length(env.state), length(env.actions), [32]),
        QNetwork(length(env.state), length(env.actions), [32]))
    try
        for gen in 1:generations
            gens += 1
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
    return gens, policy, rewards
end
