using Dates
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

function write_json(f, json)
    json = convert(Dict{String,Any}, json)
    json["time"] = Dates.format(now(), ISODateTimeFormat)
    JSON.print(f, json)
    write(f, "\n")
    flush(f)
end

function manual_command()
    manual_command_input = ""
    while length(strip(manual_command_input)) == 0
        print("Command: ")
        manual_command_input = strip(readline(stdin))
    end
end

function run()
    socket = connect(SOCKET_FILE)
    open(LOG_FILE, "a") do log_file
        while true
            sts_state = JSON.parse(readline(socket))
            write_json(log_file, Dict("sts_state" => sts_state))
            ac = agent_command(sts_state)
            if isnothing(ac)
                println("Agent gave no command. You may enter a manual command.")
                mc = manual_command()
                write_json(log_file, Dict("manual_command" => mc))
                write(socket, mc * "\n")
            else
                if typeof(ac) == String; ac = Command(ac) end
                if isnothing(extra(ac))
                    write_json(log_file, Dict("agent_command" => command(ac)))
                else
                    write_json(log_file, Dict("agent_command" => command(ac), "extra" => extra(ac)))
                end
                write(socket, command(ac) * "\n")
            end
        end
    end
end

struct Command
    command
    extra_json
end
Command(c) = Command(c, nothing)
command(c::Command) = c.command
extra(c::Command) = c.extra_json

shop_floors = []
error_streak = 0

function agent_command(state)
    global error_streak
    if "error" in keys(state)
        error_streak += 1
        sleep(1)
        return error_streak % 2 == 0 ? "wait 100" : "state"
    else
        error_streak = 0
    end
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
                min_hp = minimum(map(m -> m[1]["current_hp"], attackable_monsters))
                min_hp_monsters = filter(m -> m[1]["current_hp"] == min_hp, attackable_monsters)
                random_monster_to_attack_index = sample(min_hp_monsters)[2]
                return "play $random_card_to_play_index $random_monster_to_attack_index"
            end
            return "play $random_card_to_play_index"
        end
        if gs["screen_type"] == "COMBAT_REWARD"
            if !in("choose", state["available_commands"])
                return "proceed"
            end
            if gs["screen_state"]["rewards"][1]["reward_type"] == "POTION" && all(p -> p["id"] != "Potion Slot", gs["potions"])
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
            if !in("choose", state["available_commands"])
                return "leave"
            end
            random_shop_choice = sample(0:length(gs["choice_list"])-1)
            return "choose $random_shop_choice"
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
            random_choice = sample(0:length(gs["screen_state"]["rest_options"])-1)
            return "choose $random_choice"
        end
        if gs["screen_type"] == "CHEST"
            if !in("choose", state["available_commands"])
                return "proceed"
            end
            return "choose 0"
        end
        if gs["screen_type"] == "BOSS_REWARD"
            random_choice = sample(0:2)
            return "choose $random_choice"
        end
    end
    nothing
end

run()
