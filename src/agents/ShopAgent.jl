export ShopAgent, action, train!

struct ShopAgent
    last_visited_shop_floor :: Union{Nothing, Int}
end
ShopAgent() = ShopAgent(nothing)

function action(agent::ShopAgent, ra::RootAgent, sts_state, handled)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "SHOP_ROOM"
            if gs["floor"] == agent.last_visited_shop_floor
                return (true, "proceed")
            end
            agent.last_visited_shop_floor = nothing
            return (true, "choose 0")
        end
        if gs["screen_type"] == "SHOP_SCREEN"
            agent.last_visited_shop_floor = gs["floor"]
            if !in("choose", sts_state["available_actions"])
                return (true, "leave")
            end
            random_shop_choice = sample(0:length(gs["choice_list"])-1)
            return "choose $random_shop_choice"
        end
    end
    (false, nothing)
end

function train!(agent::ShopAgent, ra::RootAgent)
end
