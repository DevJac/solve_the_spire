using Flux
using OpenAIGym

struct PolicyNetwork{N}
    network :: N
end

Flux.@functor PolicyNetwork

function PolicyNetwork(in, out, hidden)
    layers = Any[Dense(in, hidden[1], mish, initW=Flux.kaiming_uniform)]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], mish, initW=Flux.kaiming_uniform))
    end
    push!(layers, Dense(hidden[end], out, identity, initW=Flux.kaiming_uniform))
    push!(layers, softmax)
    PolicyNetwork(Chain(layers...))
end

function (p::PolicyNetwork)(s)
    p.network(s)
end

struct Policy <: AbstractPolicy
    a_to_i
    i_to_a
    network
end

function Reinforce.action(policy::Policy, r, s, A)
    a_p = policy.network(s)
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

function optimize!(env, policy, sars, epochs=100, ϵ=0.2)
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
    target_network = deepcopy(policy.network)
    target_a_p = sum(target_network(s) .* a, dims=1)
    opt = RMSProp(0.000_1)
    for epoch in 1:epochs
        grads = gradient(params(policy.network)) do
            online_a_p = sum(policy.network(s) .* a, dims=1)
            a_ratio = online_a_p ./ target_a_p
            a_ratio_clamped = clamp.(a_ratio, 1 - ϵ, 1 + ϵ)
            -mean(minimum([a_ratio .* q; a_ratio_clamped .* q], dims=1))
        end
        Flux.Optimise.update!(opt, params(policy.network), grads)
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
    policy = Policy(x -> x+1, x -> x-1, PolicyNetwork(length(env.state), length(env.actions), [32]))
    try
        for gen in 1:generations
            sars, rs = run_episodes(env, policy, 100)
            append!(rewards, rs)
            optimize!(env, policy, sars)
            println(mean(rs))
        end
    catch e
        if !isa(e, InterruptException); rethrow() end
    finally
        close(env)
    end
    return policy, rewards, sars
end
