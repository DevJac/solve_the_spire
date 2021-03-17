export CampfireAgent, action, train!

struct CampfireAgent
end

function action(agent::CampfireAgent, ra::RootAgent, sts_state, handled)
    (false, nothing)
end

function train!(agent::CampfireAgent, ra::RootAgent)
end
