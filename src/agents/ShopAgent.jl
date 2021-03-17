export ShopAgent, action, train!

struct ShopAgent
end

function action(agent::ShopAgent, ra::RootAgent, sts_state, handled)
    (false, nothing)
end

function train!(agent::ShopAgent, ra::RootAgent)
end
