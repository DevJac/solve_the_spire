using JSON
using Sockets
using StatsBase

const SOCKET_FILE = "/home/devjac/Code/julia/solve_the_spire/relay_socket"
const LOG_FILE = "/home/devjac/Code/julia/solve_the_spire/log.txt"

function hide_map(state)
    result = deepcopy(state)
    if "map" in keys(result)
        result["map"] = "..."
    end
    if "game_state" in keys(result)
        if "map" in keys(result["game_state"])
            result["game_state"]["map"] = "..."
        end
    end
    result
end

function run()
    socket = connect(SOCKET_FILE)
    open(LOG_FILE, "a") do log_file
        while true
            sts_state = JSON.parse(readline(socket))
            JSON.print(log_file, sts_state)
            write(log_file, "\n")
            JSON.print(stdout, hide_map(sts_state), 12)
            c = command(sts_state)
            if isnothing(c)
                JSON.print(stdout, hide_map(sts_state), 12)
                print("Command: ")
                cli_input = ""
                while length(strip(cli_input)) == 0
                    cli_input = strip(readline(stdin))
                end
                write(log_file, "Command: $cli_input\n")
                write(socket, cli_input * "\n")
            else
                write(socket, c * "\n")
            end
        end
    end
end

shop_floors = []

function command(state)
    if "in_game" in keys(state) && !state["in_game"]
        return "start silent"
    end
    if "game_state" in keys(state)
        gs = state["game_state"]
        for (potion_index, potion) in enumerate(gs["potions"])
            potion_index -= 1
            if potion["can_use"]
                if potion["requires_target"]
                    monsters = collect(zip(gs["combat_state"]["monsters"], 0:100))
                    attackable_monsters = filter(m -> !m[1]["is_gone"], monsters)
                    random_monster_to_attack_index = sample(attackable_monsters)[2]
                    return "potion use $potion_index $random_monster_to_attack_index"
                else
                    return "potion use $potion_index"
                end
            end
        end
        if gs["screen_type"] == "EVENT"
            chooseables = filter(c -> "choice_index" in keys(c), gs["screen_state"]["options"])
            random_event_selection = sample(map(c -> c["choice_index"], chooseables))
            return "choose $random_event_selection"
        end
        if gs["screen_type"] == "MAP"
            random_map_selection = sample(0:length(gs["choice_list"])-1)
            return "choose $random_map_selection"
        end
        if gs["screen_type"] == "NONE"
            cs = gs["combat_state"]
            hand = collect(zip(cs["hand"], 1:100))
            monsters = collect(zip(cs["monsters"], 0:100))
            playable_hand = filter(c -> c[1]["is_playable"], hand)
            if length(playable_hand) == 0
                return "end"
            end
            attackable_monsters = filter(m -> !m[1]["is_gone"], monsters)
            random_card_to_play = sample(playable_hand)
            random_card_to_play_index = random_card_to_play[2]
            if random_card_to_play[1]["has_target"]
                random_monster_to_attack_index = sample(attackable_monsters)[2]
                return "play $random_card_to_play_index $random_monster_to_attack_index"
            end
            return "play $random_card_to_play_index"
        end
        if gs["screen_type"] == "COMBAT_REWARD"
            if !in("choose", state["available_commands"])
                return "proceed"
            end
            return "choose 0"
        end
        if gs["screen_type"] == "CARD_REWARD"
            if !in("choose", state["available_commands"])
                return "proceed"
            end
            return "choose 0"
        end
        if gs["screen_type"] == "HAND_SELECT"
            if !in("choose", state["available_commands"])
                return "proceed"
            end
            random_card_choice = sample(0:length(gs["choice_list"])-1)
            return "choose $random_card_choice"
        end
        # Peek at the shop to help us learn cards and relics.
        if gs["screen_type"] == "SHOP_ROOM"
            if gs["floor"] in shop_floors
                return "proceed"
            end
            return "choose 0"
        end
        if gs["screen_type"] == "SHOP_SCREEN"
            push!(shop_floors, gs["floor"])
            return "leave"
        end
        if gs["screen_type"] == "GRID"
            if !in("choose", state["available_commands"])
                return "confirm"
            end
            random_choice = sample(0:length(gs["choice_list"])-1)
            return "choose $random_choice"
        end
        if gs["screen_type"] == "GAME_OVER"
            return "proceed"
        end
        if gs["screen_type"] == "REST"
            if !in("choose", state["available_commands"])
                return "proceed"
            end
            return "choose 1"
        end
        if gs["screen_type"] == "CHEST"
            if !in("choose", state["available_commands"])
                return "proceed"
            end
            return "choose 0"
        end
    end
    nothing
end

run()
