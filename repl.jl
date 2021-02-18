using JSON
using Sockets

const SOCKET_FILE = "/home/devjac/Code/julia/solve_the_spire/relay_socket"

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

socket = connect(SOCKET_FILE)
while true
    sts_state = JSON.parse(readline(socket))
    JSON.print(stdout, hide_map(sts_state), 12)
    print("Command: ")
    command = readline(stdin)
    write(socket, command * "\n")
end
