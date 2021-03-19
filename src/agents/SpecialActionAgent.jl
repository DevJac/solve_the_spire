export SpecialActionAgent, action, train!

struct SpecialActionAgent
end

function action(agent::SpecialActionAgent, ra::RootAgent, sts_state, handled)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "HAND_SELECT"
            if !in("choose", state["available_commands"])
                return "proceed"
            end
            random_card_choice = sample(0:length(gs["choice_list"])-1)
            return "choose $random_card_choice"
        end
    end
    nothing
end

function train!(agent::SpecialActionAgent, ra::RootAgent)
end
