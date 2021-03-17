export EventAgent, action, train!

struct EventAgent
end

function action(agent::EventAgent, ra::RootAgent, sts_state, handled)
    (false, nothing)
end

function train!(agent::EventAgent, ra::RootAgent)
end
