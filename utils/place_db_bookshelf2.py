import os
from operator import itemgetter
import sys

sys.path.append('../benchmark')

# Macro dict (macro id -> name, x, y)

def read_node_file(fopen, benchmark):
    node_info = {}
    node_info_raw_id_name = {}
    node_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t"):
            continue
        line = line.strip().split()
        if line[-1] != "terminal":
            continue
        node_name = line[0]
        x = int(line[1])
        y = int(line[2])
        node_info[node_name] = {"id": node_cnt, "x": x, "y": y}
        node_info_raw_id_name[node_cnt] = node_name
        node_cnt += 1
    print("[place_DB]: len node_info       = ", len(node_info))
    return node_info, node_info_raw_id_name


def read_net_file(fopen, node_info):
    net_info = {}
    net_name = None
    net_cnt = 0
    for line in fopen.readlines():
        if not line.startswith("\t") and not line.startswith("NetDegree"):
            continue
        line = line.strip().split()
        if line[0] == "NetDegree":
            net_name = line[-1]
        else:
            node_name = line[0]
            if node_name in node_info:
                if not net_name in net_info:
                    net_info[net_name] = {}
                    net_info[net_name]["nodes"] = {}
                    net_info[net_name]["ports"] = {}
                if not node_name in net_info[net_name]["nodes"]:
                    x_offset = float(line[-2])
                    y_offset = float(line[-1])
                    net_info[net_name]["nodes"][node_name] = {}
                    net_info[net_name]["nodes"][node_name] = {"x_offset": x_offset, "y_offset": y_offset}
    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]['id'] = net_cnt
        net_cnt += 1
    print(f"[place_DB]: adjust net size      = {len(net_info)}")
    return net_info


def get_comp_hpwl_dict(node_info, net_info):
    comp_hpwl_dict = {}
    for net_name in net_info:
        max_idx = 0
        for node_name in net_info[net_name]["nodes"]:
            max_idx = max(max_idx, node_info[node_name]["id"])
        if not max_idx in comp_hpwl_dict:
            comp_hpwl_dict[max_idx] = []
        comp_hpwl_dict[max_idx].append(net_name)
    return comp_hpwl_dict


def get_node_to_net_dict(node_info, net_info):
    node_to_net_dict = {}
    for node_name in node_info:
        node_to_net_dict[node_name] = set()
    for net_name in net_info:
        for node_name in net_info[net_name]["nodes"]:
            node_to_net_dict[node_name].add(net_name)
    return node_to_net_dict


def get_port_to_net_dict(port_info, net_info):
    port_to_net_dict = {}
    for port_name in port_info:
        port_to_net_dict[port_name] = set()
    for net_name in net_info:
        for port_name in net_info[net_name]["ports"]:
            port_to_net_dict[port_name].add(net_name)
    return port_to_net_dict


def read_pl_file(fopen, node_info):
    max_height = 0
    max_width = 0
    for line in fopen.readlines():
        if not line.startswith('o'):
            continue
        line = line.strip().split()
        node_name = line[0]
        if not node_name in node_info:
            continue
        place_x = int(line[1])
        place_y = int(line[2])
        max_height = max(max_height, node_info[node_name]["x"] + place_x)
        max_width = max(max_width, node_info[node_name]["y"] + place_y)
        node_info[node_name]["raw_x"] = place_x
        node_info[node_name]["raw_y"] = place_y
    return max(max_height, max_width), max(max_height, max_width)


def get_node_id_to_name(node_info, node_to_net_dict):
    node_name_and_num = []
    for node_name in node_info:
        node_name_and_num.append((node_name, len(node_to_net_dict[node_name])))
    node_name_and_num = sorted(node_name_and_num, key=itemgetter(1), reverse=True)
    print("[place_DB]: node_name_and_num        = ", node_name_and_num)
    node_id_to_name = [node_name for node_name, _ in node_name_and_num]
    for i, node_name in enumerate(node_id_to_name):
        node_info[node_name]["id"] = i
    return node_id_to_name


def rank_macros(placedb, rank_mode: int = 1):
    node_id_ls = list(placedb.node_info.keys()).copy()
    for node_id in node_id_ls:
        placedb.node_info[node_id]["area"] = placedb.node_info[node_id]["x"] * placedb.node_info[node_id]["y"]
    if placedb.benchmark == "bigblue2" or placedb.benchmark == "bigblue4":
        node_id_ls.sort(key=lambda x: -placedb.node_info[x]["area"])
        return node_id_ls

    net_id_ls = list(placedb.net_info.keys()).copy()
    for net_id in net_id_ls:
        sum = 0
        for node_id in placedb.net_info[net_id]["nodes"].keys():
            sum += placedb.node_info[node_id]["area"]
        placedb.net_info[net_id]["area"] = sum
    for node_id in node_id_ls:
        placedb.node_info[node_id]["area_sum"] = 0
        for net_id in net_id_ls:
            if node_id in placedb.net_info[net_id]["nodes"].keys():
                placedb.node_info[node_id]["area_sum"] += placedb.net_info[net_id]["area"]
    if rank_mode == 1:
        node_id_ls.sort(key=lambda x: (- placedb.node_info[x]["area"], - placedb.node_info[x]["area_sum"]))
    else:
        assert rank_mode == 2
        node_id_ls.sort(key=lambda x: placedb.node_info[x]["area_sum"], reverse=True)
    return node_id_ls


def rank_macros_by_longest_chain_with_largest_area_start(placedb):
    node_id_ls = list(placedb.node_info.keys()).copy()

    for node_id in node_id_ls:
        placedb.node_info[node_id]["area"] = placedb.node_info[node_id]["x"] * placedb.node_info[node_id]["y"]

    start_node = max(node_id_ls, key=lambda x: placedb.node_info[x]["area"])

    sorted_nodes = [start_node]
    placed_nodes = {start_node}

    while len(sorted_nodes) < len(node_id_ls):
        current_node = sorted_nodes[-1]
        max_overlap = 0
        max_area = 0
        next_node = None

        for node_id in node_id_ls:
            if node_id not in placed_nodes:
                overlap_count = len(placedb.node_to_net_dict[current_node] & placedb.node_to_net_dict[node_id])
                area = placedb.node_info[node_id]["area"]

                if overlap_count > max_overlap or (overlap_count == max_overlap and area > max_area):
                    max_overlap = overlap_count
                    max_area = area
                    next_node = node_id

        if next_node:
            sorted_nodes.append(next_node)
            placed_nodes.add(next_node)
        else:
            break

    return sorted_nodes


def find_chains(node_to_net_dict):
    """
    判断模块之间有多少个链，多少个是单独的。

    :param node_to_net_dict: dict, {node_name: set_of_net_ids} - 模块与网的关系字典
    :return: chains (list of lists), single_modules (list of single modules)
    """
    visited = set()
    chains = []
    single_modules = []

    for node in node_to_net_dict:
        if node not in visited:
            chain = []
            dfs(node, node_to_net_dict, visited, chain)

            if len(chain) == 1:
                single_modules.append(chain[0])
            else:
                chains.append(chain)

    return chains, single_modules


def dfs(node, node_to_net_dict, visited, chain):
    """
    深度优先搜索 (DFS)，将所有与当前模块连接的模块添加到链中。

    :param node: 当前模块
    :param node_to_net_dict: 模块到网的关系字典
    :param visited: 已访问的模块集合
    :param chain: 当前链
    """
    visited.add(node)
    chain.append(node)

    for neighbor in node_to_net_dict[node]:
        for other_node in node_to_net_dict:
            if other_node != node and other_node not in visited:
                if node_to_net_dict[other_node] & node_to_net_dict[node]:
                    dfs(other_node, node_to_net_dict, visited, chain)


class PlaceDB:

    def __init__(self, benchmark="adaptec1"):
        self.benchmark = benchmark

        self.benchmark_DIR = "benchmark/ispd05/" + benchmark
        node_file = open(os.path.join(self.benchmark_DIR, benchmark + ".nodes"), "r")
        self.node_info, self.node_info_raw_id_name = read_node_file(node_file, benchmark)
        self.port_info = {}
        self.node_cnt = len(self.node_info)
        node_file.close()
        net_file = open(os.path.join(self.benchmark_DIR, benchmark + ".nets"), "r")
        self.net_info = read_net_file(net_file, self.node_info)
        self.net_cnt = len(self.net_info)
        net_file.close()
        pl_file = open(os.path.join(self.benchmark_DIR, benchmark + ".pl"), "r")
        self.max_height, self.max_width = read_pl_file(pl_file, self.node_info)
        pl_file.close()
        self.port_to_net_dict = {}
        self.node_to_net_dict = get_node_to_net_dict(self.node_info, self.net_info)
        self.node_id_to_name = rank_macros_by_longest_chain_with_largest_area_start(self)
        _, self.single_node = find_chains(self.node_to_net_dict)

    def debug_str(self):
        print("node_cnt = {}".format(len(self.node_info)))
        print("net_cnt = {}".format(len(self.net_info)))
        print("max_height = {}".format(self.max_height))
        print("max_width = {}".format(self.max_width))


if __name__ == "__main__":
    # ariane adaptec bigblue
    placedb = PlaceDB("adaptec2")
    placedb.debug_str()
    print(f"All: {len(placedb.node_id_to_name)}: {placedb.node_id_to_name}")
    # 调用函数
    chains, single_modules = find_chains(placedb.node_to_net_dict)

    # 输出结果
    print(f"Chain number: {len(chains)}")
    for chain in chains:
        print(f"Chain: {len(chain)}: {chain}")
    print(f"Single Modules: {len(single_modules)}: {single_modules}")

