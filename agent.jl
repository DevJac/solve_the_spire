using AgentCommands
using BSON
using Dates
using Encoders
using JSON
using Networks
using NNlib
using OwnTime
using Printf
using Profile
using SARSM
using Sockets
using StatsBase
using STSAgents
using TensorBoardLogger
using Utils

const LOG_FILE = "/home/devjac/Code/julia/solve_the_spire/log.txt"

function launch_sts()
    ENV["STS_COMMUNICATION_SOCKET"] = tempname()
    run(pipeline(`./launch_sts.sh`, stdout="sts_out.txt", stderr="sts_err.txt"), wait=false)
    timeout_start = time()
    while true
        try
            return connect(ENV["STS_COMMUNICATION_SOCKET"])
        catch e
            if typeof(e) == Base.IOError && timeout_start + 30 > time()
                sleep(3)
                continue
            end
            rethrow()
        end
    end
    throw("Couldn't connect to relay socket")
end

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

function is_game_over(sts_state)
    "game_state" in keys(sts_state) &&
        "screen_type" in keys(sts_state["game_state"]) &&
        sts_state["game_state"]["screen_type"] == "GAME_OVER"
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

function main()
    while true
        try
            mkpath("models")
            root_agent = load_root_agent()
            agent_main(root_agent)
        catch e
            if typeof(e) == ErrorException && occursin("Unexpected end of input", e.msg)
                @warn "STS crashed, will restart" exception=e
                sleep(3)
                continue
            end
            rethrow()
        end
    end
end

function agent_main(root_agent)
    socket = launch_sts()
    socket_channel = Channel(1000)
    @async begin
        while true
            put!(socket_channel, readline(socket))
        end
    end
    open(LOG_FILE, "a") do log_file
        while true
            if root_agent.ready_to_train
                train!(root_agent)
                root_agent.generation += 1
                BSON.bson(
                    @sprintf("models/agent.t.%04d.bson", max_file_number("models", "agent")+1),
                    model=root_agent)
            end
            local sts_state
            while true
                sts_state = JSON.parse(take!(socket_channel))
                if isempty(socket_channel); break end
            end
            write_json(log_file, Dict("sts_state" => sts_state))
            ac = agent_command(sts_state)
            if isnothing(ac)
                println("Agent gave no command. You may enter a manual command.")
                mc = manual_command()
                write_json(log_file, Dict("manual_command" => mc))
                write(socket, mc * "\n")
            else
                if typeof(ac) == String; ac = Command(ac) end
                if isnothing(ac.extra)
                    write_json(log_file, Dict("agent_command" => ac.command))
                else
                    write_json(log_file, Dict("agent_command" => ac.command, "extra" => ac.extra))
                end
                write(socket, ac.command * "\n")
            end
            if is_game_over(sts_state)
                root_agent.games += 1
                root_agent.ready_to_train = root_agent.games % 20 == 0
                if root_agent.games % 5 == 0 || root_agent.ready_to_train
                    BSON.bson(
                        @sprintf("models/agent.s.%04d.bson", max_file_number("models", "agent")+1),
                        model=root_agent)
                end
            end
        end
    end
end

function load_root_agent()
    if max_file_number("models", "root_agent") == 0
        RootAgent()
    else
        BSON.load(@sprintf("models/root_agent.%04d.bson", max_file_number("models", "root_agent")))[:model]
    end
end

mutable struct RootAgent
    games          :: Int
    generation     :: Int
    ready_to_train :: Bool
    tb_log
    agents
end

function RootAgent()
    tb_log = TBLogger("tb_logs/agent", tb_append)
    set_step!(tb_log, maximum(TensorBoardLogger.steps(tb_log)))
    agents = []
    RootAgent(0, 0, false, tb_log, agents)
end

function agent_command(root_agent::RootAgent, sts_state)
    increment_step!(root_agent.tb_log, 1)
    resulting_command = nothing
    for agent in root_agent.agents
        command = action(agent, root_agent, sts_state, isnothing(resulting_command))
        if !isnothing(command) && !isnothing(resulting_command)
            @error "Two agents issued commands" resulting_command command
            throw("Two agents issued commands")
        end
        resulting_command = command
    end
    resulting_command
end

function train!(root_agent::RootAgent)
    for agent in root_agent.agents
        train!(agent, root_agent)
    end
end





#shop_floors = []
#error_streak = 0
#generation_floors_reached = Int[]
#
#function agent_command(state)
    #increment_step!(tb_log, 1)
    #if "error" in keys(state)
        #global error_streak += 1
        #sleep(1)
        #return error_streak % 2 == 0 ? "wait 100" : "state"
    #else
        #global error_streak = 0
    #end
    #if "in_game" in keys(state) && !state["in_game"]
        #if length(card_playing_agent.sars.rewards) >= 2000
            #if !isempty(generation_floors_reached)
                #mean_reward = mean(x -> x[1], card_playing_agent.sars.rewards)
                #BSON.bson(
                    #@sprintf("models/cpa.%03d.bson", max_file_number("models", "cpa")+1),
                    #model=card_playing_agent, performance=mean_reward)
                #log_value(tb_log, "performance/mean_reward", mean_reward)
                #log_histogram(tb_log, "generation_floors_reached", generation_floors_reached)
                #log_text(tb_log, "generation_floors_reached_txt", repr(generation_floors_reached))
                #empty!(generation_floors_reached)
            #end
            #Profile.init(1_000_000, 0.1)
            #Profile.clear()
            #@profile train!(card_playing_agent)
            #open("profile.txt", "w") do f
                #show(f, owntime(stackframe_filter=filecontains(pwd())))
                #show(f, totaltime(stackframe_filter=filecontains(pwd())))
            #end
            #@assert length(card_playing_agent.sars.rewards) == 0
            #BSON.bson(
                #@sprintf("models/cpa.%03d.bson", max_file_number("models", "cpa")+1),
                #model=card_playing_agent)
        #end
        #empty!(shop_floors)
        #return "start silent"
    #end
    #if "game_state" in keys(state)
        #gs = state["game_state"]
        #log_value(tb_log, "rewards_length", length(card_playing_agent.sars.rewards))
        #for (potion_index, potion) in enumerate(gs["potions"])
            #potion_index -= 1
            #if potion["can_use"]
                #if potion["requires_target"]
                    #monsters = collect(zip(gs["combat_state"]["monsters"], 0:100))
                    #attackable_monsters = filter(m -> !m[1]["is_gone"], monsters)
                    #random_monster_to_attack_index = sample(attackable_monsters)[2]
                    #return "potion use $potion_index $random_monster_to_attack_index"
                #else
                    #return "potion use $potion_index"
                #end
            #end
        #end
        #if gs["screen_type"] == "EVENT"
            #chooseables = filter(c -> "choice_index" in keys(c), gs["screen_state"]["options"])
            #random_event_selection = sample(map(c -> c["choice_index"], chooseables))
            #return "choose $random_event_selection"
        #end
        #if gs["screen_type"] == "MAP"
            #reward(card_playing_agent, state, 0)
            #random_map_selection = sample(0:length(gs["choice_list"])-1)
            #return "choose $random_map_selection"
        #end
        if gs["screen_type"] == "NONE"
            cs = gs["combat_state"]
            if !any(c -> c["is_playable"], cs["hand"]); return "end" end
            reward(card_playing_agent, state)
            card_to_play = action(card_playing_agent, state)
            card_to_play_index = card_to_play[1]
            if card_to_play[2]["has_target"]
                monsters = collect(enumerate(cs["monsters"]))
                attackable_monsters = filter(m -> !m[2]["is_gone"], monsters)
                min_hp = minimum(map(m -> m[2]["current_hp"], attackable_monsters))
                min_hp_monsters = filter(m -> m[2]["current_hp"] == min_hp, attackable_monsters)
                monster_to_attack_index = sample(min_hp_monsters)[1]-1
                return "play $card_to_play_index $monster_to_attack_index"
            end
            return "play $card_to_play_index"
        end
        #if gs["screen_type"] == "COMBAT_REWARD"
            #reward(card_playing_agent, state, 0)
            #if !in("choose", state["available_commands"])
                #return "proceed"
            #end
            #if gs["screen_state"]["rewards"][1]["reward_type"] == "POTION" && all(p -> p["id"] != "Potion Slot", gs["potions"])
                #return "proceed"
            #end
            #return "choose 0"
        #end
        #if gs["screen_type"] == "CARD_REWARD"
            #if !in("choose", state["available_commands"])
                #return "proceed"
            #end
            #return "choose 0"
        #end
        #if gs["screen_type"] == "HAND_SELECT"
            #if !in("choose", state["available_commands"])
                #return "proceed"
            #end
            #random_card_choice = sample(0:length(gs["choice_list"])-1)
            #return "choose $random_card_choice"
        #end
        #if gs["screen_type"] == "SHOP_ROOM"
            #if gs["floor"] in shop_floors
                #return "proceed"
            #end
            #return "choose 0"
        #end
        #if gs["screen_type"] == "SHOP_SCREEN"
            #push!(shop_floors, gs["floor"])
            #if !in("choose", state["available_commands"])
                #return "leave"
            #end
            #random_shop_choice = sample(0:length(gs["choice_list"])-1)
            #return "choose $random_shop_choice"
        #end
        #if gs["screen_type"] == "GRID"
            #if !in("choose", state["available_commands"])
                #return "confirm"
            #end
            #random_choice = sample(0:length(gs["choice_list"])-1)
            #return "choose $random_choice"
        #end
        #if gs["screen_type"] == "GAME_OVER"
            #log_value(tb_log, "performance/floor_reached", gs["floor"])
            #push!(generation_floors_reached, gs["floor"])
            #reward(card_playing_agent, state, 0)
            #return "proceed"
        #end
        #if gs["screen_type"] == "REST"
            #if !in("choose", state["available_commands"])
                #return "proceed"
            #end
            #random_choice = sample(0:length(gs["screen_state"]["rest_options"])-1)
            #return "choose $random_choice"
        #end
        #if gs["screen_type"] == "CHEST"
            #if !in("choose", state["available_commands"])
                #return "proceed"
            #end
            #return "choose 0"
        #end
        #if gs["screen_type"] == "BOSS_REWARD"
            #random_choice = sample(0:2)
            #return "choose $random_choice"
        #end
    #end
    #nothing
#end

main()
