using Sockets

const SOCKET_FILE = "/home/devjac/Code/julia/solve_the_spire/relay_socket"

server = listen(SOCKET_FILE)
println("ready")
socket = accept(server)

last_command = time()

sts_to_socket = @async begin
    while true
        write(socket, readline(stdin) * "\n")
    end
end

socket_to_sts = @async begin
    while true
        command = readline(socket)
        sleep(0.01)
        global last_command = time()
        write(stdout, command * "\n")
    end
end

state_wakeup = @async begin
    while true
        sleep(10)
        if last_command + 25 < time()
            write(stdout, "state\n")
        end
    end
end

wait(sts_to_socket)
wait(socket_to_sts)
wait(state_wakeup)
