using JSON
using Sockets

const SOCKET_FILE = "/home/devjac/Code/julia/solve_the_spire/relay_socket"

socket = connect(SOCKET_FILE)
while true
    sts_state = JSON.parse(readline(socket))
    JSON.print(stdout, sts_state, 8)
    print("Command: ")
    command = readline(stdin)
    write(socket, command * "\n")
end
