using Sockets

println("ready")

const SOCKET_FILE = ENV["STS_COMMUNICATION_SOCKET"]
const server = listen(SOCKET_FILE)
const socket = accept(server)

last_command = time()

@async begin
    while true
        sleep(10)
        if last_command + 10 < time()
            write(stdout, "state\n")
        end
    end
end

@async begin
    while true
        write(socket, readline(stdin) * "\n")
    end
end

wait(@async begin
    while true
        command = readline(socket)
        if command == ""; break end
        global last_command = time()
        write(stdout, command * "\n")
    end
end)
