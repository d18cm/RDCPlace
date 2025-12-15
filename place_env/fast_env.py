import gym
import math
from typing import Dict, Tuple, List
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt, patches
from utils.comp_res import comp_res

class PlaceEnvGpu(gym.Env):
    def __init__(
            self,
            placedb,
            num_macros_to_place: int,
            grid: int,
            reward_scale: float = 200.0,
            device: str = "cuda",
    ) -> None:
        super().__init__()
        self.placedb = placedb
        self.num_macros: int = placedb.node_cnt
        self.num_nets: int = placedb.net_cnt
        self.node_name_list: List[str] = placedb.node_id_to_name

        self.grid: int = grid
        self.max_height: float = placedb.max_height
        self.max_width: float = placedb.max_width
        self.ratio: float = self.max_height / self.grid
        print(f"[Env] : self.max_height       {self.max_height}")
        print(f"[Env] : self.ratio       {self.ratio}")

        self.size_x: List[int] = [max(1, math.ceil(placedb.node_info[name]['x'] / self.ratio))
                                  for name in self.node_name_list]
        self.size_y: List[int] = [max(1, math.ceil(placedb.node_info[name]['y'] / self.ratio))
                                  for name in self.node_name_list]

        self.num_macros_to_place = num_macros_to_place
        self.wire_mask_scale = self.grid * self.grid
        self.reward_scale = float(reward_scale)
        self.chain_broken = False

        self.device = torch.device(device)
        self._rows = torch.arange(self.grid, device=self.device, dtype=torch.float32).unsqueeze(1)  # [G,1]
        self._cols = torch.arange(self.grid, device=self.device, dtype=torch.float32).unsqueeze(1)  # [G,1]

        self.cum_reward: float = 0.0
        self.t: int = 0
        self.node_pos: Dict[str, Tuple[int, int, int, int]] = {}
        self.net_bound_info: Dict[str, Dict[str, int]] = {}
        self.state: torch.Tensor = torch.zeros((3, self.grid, self.grid), device=self.device,
                                               dtype=torch.float32)  # [3, G, G]

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        self.t = 0
        self.node_pos.clear()
        self.net_bound_info.clear()
        self.cum_reward = 0.0

        canvas = torch.zeros((self.grid, self.grid), device=self.device, dtype=torch.float32)
        wire_mask = torch.zeros_like(canvas)

        next_x = self.size_x[0]
        next_y = self.size_y[0]
        position_mask = self._calc_position_mask(canvas, next_x, next_y)

        self.state = torch.stack([canvas, wire_mask, position_mask], dim=0)
        return self.state

    @torch.no_grad()
    def step(self, action: int):
        assert 0 <= action < self.grid * self.grid, "action out of range"

        canvas, wire_mask, position_mask = self.state[0], self.state[1], self.state[2]

        x = int(torch.div(action, self.grid, rounding_mode='floor'))
        y = int(action % self.grid)
        sx = self.size_x[self.t]
        sy = self.size_y[self.t]

        # --------- 执行动作：正常奖励 ----------
        reward = - wire_mask[x, y] / self.reward_scale * self.wire_mask_scale

        # --------- 非法动作：给惩罚(正常情况不会发生，但是排除极端因素因为序列造成的) ----------
        if position_mask[x, y] >= 1.0:
            penalty = torch.as_tensor(200000.0 / self.reward_scale,
                                      device=self.device, dtype=torch.float32)
            reward = reward - penalty

        # --------- 断链动作：给惩罚 ----------
        if self.chain_broken and not self.placedb.node_id_to_name[self.t] in self.placedb.single_node:
            r = getattr(self, "chain_radius", 5)
            scale = getattr(self, "chain_penalty_scale", 1.0)

            ring_mean = self._ring_mean(canvas, x, y, sx, sy, r)
            penalty = (ring_mean * scale) / self.reward_scale
            reward = reward - penalty

            self.chain_broken = False

        canvas[x:x + sx, y:y + sy] = 1.0
        canvas[x:x + sx, y] = 0.5
        if y + sy - 1 < self.grid:
            canvas[x:x + sx, y + sy - 1] = 0.5
        canvas[x, y:y + sy] = 0.5
        if x + sx - 1 < self.grid:
            canvas[x + sx - 1, y:y + sy] = 0.5

        node_name = self.node_name_list[self.t]
        self.node_pos[node_name] = (x, y, sx, sy)

        self._update_net_bounds(node_name, x, y)

        self.t += 1
        done = self._is_done()

        if not done:
            next_x = self.size_x[self.t]
            next_y = self.size_y[self.t]
            position_mask = self._calc_position_mask(canvas, next_x, next_y)
            wire_mask = self._calc_wire_mask_for_node(self.node_name_list[self.t])
        else:
            position_mask = torch.ones_like(canvas)
            wire_mask = torch.zeros_like(canvas)
            # if need
            # hpwl, cost = comp_res(self.placedb, self.node_pos, self.ratio)

        self.state = torch.stack([canvas, wire_mask, position_mask], dim=0)

        info = {
            'placed_node_idx': self.t - 1,
            'action_idx': int(action),
            'xy': (int(x), int(y)),
            'node_name': self.node_name_list[self.t - 1],
        }
        return self.state, float(reward.item()), bool(done), info

    @torch.no_grad()
    def _calc_position_mask(self, canvas: torch.Tensor, next_x: int, next_y: int) -> torch.Tensor:
        G = self.grid

        if canvas.sum() == 0:
            mask = torch.ones_like(canvas)
            mask[:G - next_x, :G - next_y] = 0
            return mask

        occ = (canvas > 0).float().unsqueeze(0).unsqueeze(0)  # [1, 1, G, G]
        blocked = F.max_pool2d(occ, kernel_size=(next_x, next_y), stride=1, padding=0)  # [1, 1, G-next_x+1, G-next_y+1]
        mask = torch.zeros_like(canvas)
        mask[:G - next_x + 1, :G - next_y + 1] = blocked[0, 0]
        if G - next_x + 1 < G:
            mask[G - next_x + 1:, :] = 1.0
        if G - next_y + 1 < G:
            mask[:, G - next_y + 1:] = 1.0
        return mask

    @torch.no_grad()
    def _calc_wire_mask_for_node(self, node_name: str) -> torch.Tensor:
        G = self.grid
        rows = self._rows  # [G, 1]
        cols = self._cols  # [G, 1]
        dev = self.device

        nets = [n for n in self.placedb.node_to_net_dict[node_name] if n in self.net_bound_info]

        if not nets:
            self.chain_broken = True
            return torch.zeros((G, G), device=dev, dtype=torch.float32)

        # per-net 参数
        node_x_half = self.placedb.node_info[node_name]['x'] / 2.0
        node_y_half = self.placedb.node_info[node_name]['y'] / 2.0
        ratio = self.ratio

        off_x = torch.tensor([self.placedb.net_info[n]["nodes"][node_name]["x_offset"] for n in nets],
                             device=dev, dtype=torch.float32)
        off_y = torch.tensor([self.placedb.net_info[n]["nodes"][node_name]["y_offset"] for n in nets],
                             device=dev, dtype=torch.float32)
        dx = torch.round((torch.as_tensor(node_x_half, device=dev) + off_x) / ratio)  # [K]
        dy = torch.round((torch.as_tensor(node_y_half, device=dev) + off_y) / ratio)  # [K]

        min_x = torch.tensor([self.net_bound_info[n]['min_x'] for n in nets], device=dev, dtype=torch.float32)
        max_x = torch.tensor([self.net_bound_info[n]['max_x'] for n in nets], device=dev, dtype=torch.float32)
        min_y = torch.tensor([self.net_bound_info[n]['min_y'] for n in nets], device=dev, dtype=torch.float32)
        max_y = torch.tensor([self.net_bound_info[n]['max_y'] for n in nets], device=dev, dtype=torch.float32)

        w = torch.tensor([self.placedb.net_info[n].get('weight', 1.0) for n in nets], device=dev, dtype=torch.float32)

        sx = (min_x - dx).clamp_(0.0, G - 1.0)
        ex = (max_x - dx).clamp_(0.0, G - 1.0)
        sy = (min_y - dy).clamp_(0.0, G - 1.0)
        ey = (max_y - dy).clamp_(0.0, G - 1.0)

        dist_x = torch.relu(sx.unsqueeze(0) - rows) + torch.relu(rows - ex.unsqueeze(0))
        dist_y = torch.relu(sy.unsqueeze(0) - cols) + torch.relu(cols - ey.unsqueeze(0))

        row_sum = (dist_x * w.unsqueeze(0)).sum(dim=1)  # [G]
        col_sum = (dist_y * w.unsqueeze(0)).sum(dim=1)  # [G]
        net_img = row_sum.unsqueeze(1) + col_sum.unsqueeze(0)  # [G, G]

        return net_img / self.wire_mask_scale

    @torch.no_grad()
    def _update_net_bounds(self, node_name: str, x: int, y: int) -> None:
        node_x = self.placedb.node_info[node_name]['x']
        node_y = self.placedb.node_info[node_name]['y']
        ratio = self.ratio

        base_x = x * ratio
        base_y = y * ratio

        for net_name in self.placedb.node_to_net_dict[node_name]:
            px = base_x + node_x / 2.0 + self.placedb.net_info[net_name]["nodes"][node_name]["x_offset"]
            py = base_y + node_y / 2.0 + self.placedb.net_info[net_name]["nodes"][node_name]["y_offset"]
            pin_x = int(round(px / ratio))
            pin_y = int(round(py / ratio))

            if net_name in self.net_bound_info:
                b = self.net_bound_info[net_name]
                if pin_x < b['min_x']: b['min_x'] = pin_x
                if pin_x > b['max_x']: b['max_x'] = pin_x
                if pin_y < b['min_y']: b['min_y'] = pin_y
                if pin_y > b['max_y']: b['max_y'] = pin_y
            else:
                self.net_bound_info[net_name] = {
                    'min_x': pin_x, 'max_x': pin_x,
                    'min_y': pin_y, 'max_y': pin_y,
                }

    def _ring_mean(self, canvas: torch.Tensor, x: int, y: int, sx: int, sy: int, r: int = 10) -> torch.Tensor:
        assert canvas.dim() == 2, f"canvas must be [G, G], got {canvas.shape}"
        dev = canvas.device
        dtype = canvas.dtype

        pad_canvas = F.pad(canvas, (r, r, r, r), mode='constant', value=1.0)

        xp, yp = x + r, y + r

        top = pad_canvas[xp - r: xp, yp - r: yp + sy + r]
        bottom = pad_canvas[xp + sx: xp + sx + r, yp - r: yp + sy + r]
        left = pad_canvas[xp: xp + sx, yp - r: yp]
        right = pad_canvas[xp: xp + sx, yp + sy: yp + sy + r]

        ring_sum = top.sum() + bottom.sum() + left.sum() + right.sum()

        ring_area = (sx + 2 * r) * (sy + 2 * r) - (sx * sy)
        ring_mean = ring_sum / torch.as_tensor(float(ring_area), device=dev, dtype=dtype)
        return ring_mean

    def _is_done(self) -> bool:
        return (self.t >= self.num_macros) or (self.t >= self.num_macros_to_place)

    def save_fig(self, file_path):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        for node_name in self.node_pos:
            x, y, size_x, size_y = self.node_pos[node_name]
            ax1.add_patch(
                patches.Rectangle(
                    (x / self.grid, y / self.grid),  # (x,y)
                    size_x / self.grid,  # width
                    size_y / self.grid, linewidth=1, edgecolor='k',
                )
            )
        fig1.savefig(file_path, dpi=90, bbox_inches='tight')
        plt.close()
