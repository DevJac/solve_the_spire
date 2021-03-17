export PotionAgent, action, train!

struct PotionAgent
end

function action(agent::PotionAgent, ra::RootAgent, sts_state, handled)
    (false, nothing)
end

function train!(agent::PotionAgent, ra::RootAgent)
end
