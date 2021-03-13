module AgentCommands

export Command

struct Command
    command
    extra
end
Command(c) = Command(c, nothing)

end # module
