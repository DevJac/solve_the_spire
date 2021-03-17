export DraftingAgent, action, train!

struct DraftingAgent
end

function action(agent::DraftingAgent, ra::RootAgent, sts_state, handled)
    (false, nothing)
end

function train!(agent::DraftingAgent, ra::RootAgent)
end
