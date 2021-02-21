using Sockets

const SOCKET_FILE = "/home/devjac/Code/julia/solve_the_spire/relay_socket"

server = listen(SOCKET_FILE)
println("ready")
socket = accept(server)

sts_to_socket = @async begin
    while true
        write(socket, readline(stdin) * "\n")
    end
end

socket_to_sts = @async begin
    while true
        command = readline(socket)
        if command == ""; return end
        write(stdout, command * "\n")
    end
end

wait(sts_to_socket)
wait(socket_to_sts)
