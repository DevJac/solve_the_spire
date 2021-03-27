export ShopAgent, action, train!

mutable struct ShopAgent
    player_embedder
    deck_embedder
    potions_embedder
    relics_embedder
    all_card_choice_embedder
    single_card_choice_embedder
    all_relic_choice_embedder
    single_relic_choice_embedder
    all_potion_choice_embedder
    single_potion_choice_embedder
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
    all_card_choice_embedder = PoolNetwork(legnth(card_encoder)+1, 20, [50])
    single_card_choice_embedder = VanillaNetwork(length(card_encoder)+1, 20, [50])
    all_relic_choice_embedder = VanillaNetwork(length(relics_encoder), 20, [50])
    single_relic_choice_embedder = VanillaNetwork(length(relics_encoder)+1, 20, [50])
    all_potion_choice_embedder = VanillaNetwork(length(potions_encoder), 20, [50])
    single_potion_choice_embedder = VanillaNetwork(length(potions_encoder)+1, 20, [50])
    policy = VanillaNetwork(
        sum(length, [
            player_embedder,
            deck_embedder,
            potions_embedder,
            relics_embedder,
            all_card_choice_embedder,
            single_card_choice_embedder,
            all_relic_choice_embedder,
            single_relic_choice_embedder,
            all_potion_choice_embedder,
            single_potion_choice_embedder]),
        1, [200, 50, 50])
    critic = VanillaNetwork(
        sum(length, [
            player_embedder,
            deck_embedder,
            potions_embedder,
            relics_embedder,
            all_card_choice_embedder,
            all_relic_choice_embedder,
            all_potion_choice_embedder]),
        1, [200, 50, 50])
    policy_opt = ADADelta()
    critic_opt = ADADelta()
    sars = SARS()
    last_visited_shop_floor = nothing
    ShopAgent(
        player_embedder
        deck_embedder
        potions_embedder
        relics_embedder
        all_card_choice_embedder
        single_card_choice_embedder
        all_relic_choice_embedder
        single_relic_choice_embedder
        all_potion_choice_embedder
        single_potion_choice_embedder
        policy
        critic
        policy_opt
        critic_opt
        sars
        last_visited_shop_floor)
end

function action_probabilities(agent::ShopAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    local cards_offered
    local relics_offered
    local potions_offered
    local purge_offered = false
    local purge_cost = 0
    Zygote.ignore() do
        if gs["screen_type"] == "SHOP_SCREEN"
            cards_offered = gs["screen_state"]["cards"]
            relics_offered = gs["screen_state"]["relics"]
            potions_offered = gs["screen_state"]["potions"]
            purge_offered = gs["screen_state"]["purge_available"]
            purge_cost = gs["screen_state"]["purge_cost"]
        end
    end
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
end

function train!(agent::ShopAgent, ra::RootAgent)
end
