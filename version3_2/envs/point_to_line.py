def find_triangular_connections_unique(grid):
    """Find unique index connections in the triangular mesh."""
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)]
    connections = set()
    size = len(connections)
    connection_actions = []
    
    for x in range(rows):
        for y in range(cols):
            if grid[x][y] != 0:  # Only process non-zero points
                for action,(dx,dy) in enumerate(directions):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1:
                        # Sort the connection to handle repeatness
                        connection = tuple(sorted([(x, y), (nx, ny)]))
                        connections.add(connection)
                        if len(connections) > size : #append success
                            size+=1
                            # print(action)
                            connection = sorted([(x, y), (nx, ny)])
                            connection_action = connection+[action]
                            # print(connection_action)
                            connection_actions.append(connection_action)
                            # print(connection_actions)
    
    return [list(connection) for connection in connection_actions]

def merge_connection(conncections):
    merged_connections = set()
    directions = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)]
    for connection in connections:
        print("Process",connection)
        start_row = connection[0][0]
        start_column = connection[0][1]
        end_row = connection[1][0]
        end_column = connection[1][1]
        action = connection[2]

        next_start = [end_row,end_column]
        while True:    
            next_end = [end_row + directions[action][0],end_column + directions[action][1]]
            if [tuple(next_start),tuple(next_end),action] in connections:
                # print("trigger")
                conncections.remove([tuple(next_start),tuple(next_end),action])
                next_start = next_end
            else:
                break

        merged_connection = tuple(sorted([(start_row, start_column), (next_start[0], next_start[1])]))
        merged_connections.add(merged_connection)
    
    return merged_connections 

def trajectory_to_line(grid):
    connections = find_triangular_connections_unique(grid)
    merged_connections = merge_connection(connections)
    
    return merged_connections
    
if __name__ == '__main__':
    # Example Matrix
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0]
    ]

    connections = find_triangular_connections_unique(grid)
    print(connections)
    merged_connections = merge_connection(connections)
    print("Unique Connections:", merged_connections) #output a dict: {((1, 3), (2, 2)), ((1, 1), (2, 2)), ((2, 2), (3, 3)), ((1, 1), (1, 3))}
