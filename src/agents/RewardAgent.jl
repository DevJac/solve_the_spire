export RewardAgent, action, train!

struct RewardAgent
end

function action(agent::RewardAgent, ra::RootAgent, sts_state, handled)
    (false, nothing)
end

function train!(agent::RewardAgent, ra::RootAgent)
end
