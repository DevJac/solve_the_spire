using JSON
using StatsBase

function command(state)
    if "start" in state["available_commands"]
        return "start ironclad"
    end
    if "play" in state["available_commands"]
        cards_in_hand = length(state["game_state"]["combat_state"]["hand"])
        card_index_to_play = 0
        for _ in 1:100
            card_index_to_play = rand(1:cards_in_hand)
            if state["game_state"]["combat_state"]["hand"][card_index_to_play]["is_playable"]; break end
        end
        if state["game_state"]["combat_state"]["hand"][card_index_to_play]["has_target"]
            for i in 0:length(state["game_state"]["combat_state"]["monsters"])-1
                if !state["game_state"]["combat_state"]["monsters"][i+1]["is_gone"]
                    return "play $card_index_to_play $i"
                end
            end
        else
            return "play $card_index_to_play"
        end
    end
    if "proceed" in state["available_commands"]
        return "proceed"
    end
    if "confirm" in state["available_commands"]
        return "confirm"
    end
    if "end" in state["available_commands"]
        return "end"
    end
    if "leave" in state["available_commands"]
        return "leave"
    end
    if "choose" in state["available_commands"]
        return "choose 0"
    end
end

function hide_map!(d)
    if "game_state" in keys(d)
        d["game_state"]["map"] = "..."
    end
    if "map" in keys(d)
        d["map"] = "..."
    end
end

function run()
    println("ready")
    open("/home/devjac/Code/julia/solve_the_spire/out.txt", "a") do f
        try
            write(f, "\n\n\n\n============ new run ============\n")
            for t in 1:1000
                msg_raw = readline()
                msg = JSON.parse(msg_raw)
                write(f, "received state:\n")
                hide_map!(msg)
                JSON.print(f, msg, 8)
                flush(f)
                next_command = command(msg)
                write(f, "sending command: $next_command\n")
                if isnothing(next_command)
                    write(f, "done.\n")
                    break
                end
                flush(f)
                println(next_command)
            end
        catch e
            showerror(f, e, catch_backtrace())
            rethrow()
        end
    end
end

run()
