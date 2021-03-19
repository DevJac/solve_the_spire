export PotionAgent, action, train!

struct PotionAgent
end

function action(agent::PotionAgent, ra::RootAgent, sts_state, handled)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        for (potion_index, potion) in enumerate(gs["potions"])
            potion_index -= 1  # STS uses 0-based indexing
            if potion["can_use"]
                if potion["requires_target"]
                    monsters = collect(enumerate(gs["combat_state"]["monsters"]))
                    attackable_monsters = filter(m -> !m[2]["is_gone"], monsters)
                    monster_index = sample(attackable_monsters)[1]-1
                    return "potion use $potion_index $monster_index"
                else
                    return "potion use $potion_index"
                end
            end
        end
    end
    nothing
end

function train!(agent::PotionAgent, ra::RootAgent)
end
