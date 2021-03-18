export RewardAgent, action, train!

struct RewardAgent
end

function action(agent::RewardAgent, ra::RootAgent, sts_state, handled)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "COMBAT_REWARD"
            if !in("choose", state["available_commands"])
                return (true, "proceed")
            end
            if gs["screen_state"]["rewards"][1]["reward_type"] == "POTION" && all(p -> p["id"] != "Potion Slot", gs["potions"])
                return (true, "proceed")
            end
            return (true, "choose 0")
        end
        if gs["screen_type"] == "CHEST"
            if !in("choose", state["available_commands"])
                return (true, "proceed")
            end
            return (true, "choose 0")
        end
        if gs["screen_type"] == "BOSS_REWARD"
            random_choice = sample(0:2)
            return (true, "choose $random_choice")
        end
    end
    (false, nothing)
end

function train!(agent::RewardAgent, ra::RootAgent)
end
