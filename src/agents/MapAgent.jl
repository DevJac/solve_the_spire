export MapAgent, action, train!

const MAP_ROOM_TYPES = ('T', 'E', 'R', 'M', '$', '?')

struct MapAgent
end

function action(agent::MapAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "MAP"
            random_map_selection = sample(0:length(gs["choice_list"])-1)
            return "choose $random_map_selection"
        end
    end
end

function train!(agent::MapAgent, ra::RootAgent)
end

function map_path_counts(map_state, x, y)
    path_counts = StatsBase.countmap.(map_paths(map_state, x, y))
    map(MAP_ROOM_TYPES) do room_type
        (minimum(pc -> get(pc, room_type, 0), path_counts), maximum(pc -> get(pc, room_type, 0), path_counts))
    end
end

function map_paths(map_state, x, y, all_paths=Vector{Char}[], path=Char[])
    leaf_node = true
    for map_node in map_state
        if map_node["x"] == x && map_node["y"] == y
            push!(path, only(map_node["symbol"]))
            for child_node in map_node["children"]
                leaf_node = false
                map_paths(map_state, child_node["x"], child_node["y"], all_paths, deepcopy(path))
            end
            break
        end
    end
    if leaf_node
        push!(all_paths, deepcopy(path))
    end
    all_paths
end
