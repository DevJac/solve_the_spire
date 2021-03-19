export MapAgent, action, train!

struct MapAgent
end

function action(agent::MapAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "MAP"
            random_map_selection = sample(0:length(gs["choice_list"])-1)
            return "choose $random_map_selection"
        end
    end
end

function train!(agent::MapAgent, ra::RootAgent)
end
