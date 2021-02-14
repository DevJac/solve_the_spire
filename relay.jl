using Sockets

const SOCKET_FILE = "/home/devjac/Code/julia/solve_the_spire/relay_socket"

server = listen(SOCKET_FILE)
println("ready")
socket = accept(server)
while true
    write(socket, readline(stdin) * "\n")
    command = readline(socket)
    if command == ""; break end
    write(stdout, command * "\n")
end
