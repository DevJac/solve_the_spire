export ShopAgent, action, train!

mutable struct ShopAgent
    player_embedder
    deck_embedder
    potions_embedder
    relics_embedder
    card_choice_embedder
    relic_choice_embedder
    potion_choice_embedder
    policy
    critic
    policy_opt
    critic_opt
    sars
    last_visited_shop_floor :: Union{Nothing, Int}
end
function ShopAgent()
    player_embedder = VanillaNetwork(length(player_basic_encoder), length(player_basic_encoder), [50])
    deck_embedder = PoolNetwork(length(card_encoder), 20, [50])
    potions_embedder = VanillaNetwork(length(potions_encoder), 20, [50])
    relics_embedder = VanillaNetwork(length(relics_encoder), 20, [50])
    card_choice_embedder = PoolNetwork(legnth(card_encoder)+1, 20, [50])
    relic_choice_embedder = PoolNetwork(length(relics_encoder)+1, 20, [50])
    potion_choice_embedder = PoolNetwork(length(potions_encoder)+1, 20, [50])
    policy = VanillaNetwork(
        sum(length, [
            player_embedder,
            deck_embedder,
            potions_embedder,
            relics_embedder,
            card_choice_embedder,
            relic_choice_embedder,
            potion_choice_embedder])
        1, [200, 50, 50])
    critic = VanillaNetwork(
        sum(length, [
            player_embedder,
            deck_embedder,
            potions_embedder,
            relics_embedder,
            pool(card_choice_embedder),
            pool(relic_choice_embedder),
            pool(potion_choice_embedder)]),
        1, [200, 50, 50])
    policy_opt = ADADelta()
    critic_opt = ADADelta()
    sars = SARS()
    last_visited_shop_floor = nothing
    ShopAgent(
        player_embedder,
        deck_embedder,
        potions_embedder,
        relics_embedder,
        card_choice_embedder,
        relic_choice_embedder,
        potion_choice_embedder,
        policy,
        critic,
        policy_opt,
        critic_opt,
        sars,
        last_visited_shop_floor :: Union{Nothing, Int})
end

function action_probabilities(agent::ShopAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    local actions = []
    local purge_cost = 0
    local purge_offered = false
    local cards_offered = []
    local relics_offered = []
    local potions_offered = []
    local chest_available = false
    local saphire_key = false
    Zygote.ignore() do
        if gs["screen_type"] == "SHOP_SCREEN"
            purge_cost = gs["screen_state"]["purge_cost"]
            purge_offered = gs["screen_state"]["purge_available"] && gs["gold"] >= purge_cost
            cards_offered = filter(c -> c["cost"] <= gs["gold"], gs["screen_state"]["cards"])
            relics_offered = filter(r -> r["cost"] <= gs["gold"], gs["screen_state"]["relics"])
            potions_offered = filter(p -> p["cost"] <= gs["gold"], gs["screen_state"]["potions"])
        end
        if gs["screen_type"] == "BOSS_REWARD"
            relics_offered = gs["screen_state"]["relics"]
        end
        if gs["screen_type"] == "CHEST"
            chest_available = true
        end
        if gs["screen_type"] == "COMBAT_REWARD"
            relics_offered = map(x -> x["relic"], filter(x -> x["reward_type"] == "RELIC", gs["screen_state"]["rewards"]))
            potions_offered = map(x -> x["potion"], filter(x -> x["reward_type"] == "POTION", gs["screen_state"]["rewards"]))
            saphire_key = any(x -> x["reward_type"] == "SAPHIRE_KEY", gs["screen_state"]["rewards"])
        end
        function populate_cost(x)
            x["cost"] = get(x, "cost", 0)
            x
        end
        map(populate_cost, cards_offered)
        map(populate_cost, relics_offered)
        map(populate_cost, potions_offered)
    end
    player_e = agent.player_embedder(player_basic_encoder(sts_state))
    deck_e = agent.deck_embedder(card_encoder, gs["deck"])
    potions_e = agent.potions_embedder(potions_encoder(gs["potions"]))
    relics_e = agent.relics_embedder(relics_encoder(gs["relics"]))
    card_choice_e = agent.card_choice_embedder(c -> [card_encoder(c); c["cost"]], cards_offered)
    relic_choice_e = agent.relic_choice_embedder(r -> [relics_encoder([r]); r["cost"]], relics_offered)
    potion_choice_e = agent.potion_choice_embedder(p -> [potions_encoder([p]); p["cost"]], potions_offered)
    choices_e = diagcat(
        card_choice_e,
        relic_choice_e,
        potion_choice_e,
        
    )
    action_weights = agent.policy(vcat(
    ))
end





function action(agent::ShopAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "SHOP_ROOM"
            if gs["floor"] == agent.last_visited_shop_floor
                return "proceed"
            end
            agent.last_visited_shop_floor = nothing
            return "choose 0"
        end
        if gs["screen_type"] == "SHOP_SCREEN"
            agent.last_visited_shop_floor = gs["floor"]
            if !in("choose", sts_state["available_commands"])
                return "leave"
            end
            random_shop_choice = sample(0:length(gs["choice_list"])-1)
            return "choose $random_shop_choice"
        end
    end
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "COMBAT_REWARD"
            if !in("choose", sts_state["available_commands"])
                return "proceed"
            end
            if gs["screen_state"]["rewards"][1]["reward_type"] == "POTION" && all(p -> p["id"] != "Potion Slot", gs["potions"])
                return "proceed"
            end
            return "choose 0"
        end
        if gs["screen_type"] == "CHEST"
            if !in("choose", sts_state["available_commands"])
                return "proceed"
            end
            return "choose 0"
        end
        if gs["screen_type"] == "BOSS_REWARD"
            random_choice = sample(0:2)
            return "choose $random_choice"
        end
    end
end

function train!(agent::ShopAgent, ra::RootAgent)
end
