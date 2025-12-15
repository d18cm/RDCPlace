
def save_placement(file_path, placedb, node_pos, ratio):
    with open(file_path, 'w') as f:
        node_place = {}
        for node_name in node_pos:
            x, y, _, _ = node_pos[node_name]
            x = round(x * ratio + ratio)
            y = round(y * ratio + ratio)
            node_place[node_name] = (x, y)
        for node_name in placedb.node_info:
            if node_name not in node_place:
                continue
            x, y = node_place[node_name]
            f.write(f'{node_name}\t{x}\t{y}\t: \tN /FIXED\n')