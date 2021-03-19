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
    manual_command_input
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
                root_agent.ready_to_train = false
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
            ac = agent_command(root_agent, sts_state)
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
    mfn = max_file_number("models", "agent")
    if mfn == 0
        return RootAgent()
    else
        for marker in ("t", "s")
            f = @sprintf("agent.%s.%04d.bson", marker, mfn)
            if f in readdir("models")
                return BSON.load("models/" * f)[:model]
            end
        end
    end
    throw("Couldn't load root agent")
end

main()
