import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple

# ========== NamedTuple ==========
Transition = namedtuple(
    'Transition',
    ['state', 'action', 'reward', 'a_log_prob', 'next_state']
)


# ========== Actor ==========
class Actor(nn.Module):
    def __init__(self, grid=224, k_pool=11, tau=1.0):
        super().__init__()
        self.grid = grid
        self.grid2 = grid * grid
        self.tau = nn.Parameter(torch.tensor(float(tau)))

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),  # -> [B,1,G,G]
        )

        self.pool = nn.AvgPool2d(kernel_size=k_pool, stride=1, padding=k_pool // 2, count_include_pad=False)

        self.gate_mlp = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 1), nn.Sigmoid()
        )

        self.prior_fuse = nn.Conv2d(2, 1, 1, bias=True)

        self.softmax = nn.Softmax(dim=-1)

        with torch.no_grad():
            yy, xx = torch.meshgrid(
                torch.arange(grid, dtype=torch.float32),
                torch.arange(grid, dtype=torch.float32),
                indexing='ij'
            )
            dist_to_border = torch.minimum(
                torch.minimum(xx, grid - 1 - xx),
                torch.minimum(yy, grid - 1 - yy)
            ) / (grid / 2.0)
            self.register_buffer("border_prior", dist_to_border.unsqueeze(0).unsqueeze(0))  # [1,1,G,G]

    def forward(self, x):
        """
        x: [B, 3, G, G], order = [canvas, wiremask, position_mask]
        returns: [B, G*G] probs
        """
        assert x.dim() == 4 and x.size(2) == self.grid and x.size(3) == self.grid
        B = x.size(0)

        canvas = x[:, 0:1]  # [B,1,G,G]
        wire = x[:, 1:2]  # [B,1,G,G]
        posmsk = x[:, 2:3]  # [B,1,G,G]

        logits_data = self.cnn(x)  # [B,1,G,G]

        local_occ = self.pool(canvas)  # [B,1,G,G], 0~1
        prior_space = 1.0 - torch.clamp(local_occ, 0, 1)

        # Boundary prior
        prior_border = self.border_prior.expand(B, -1, -1, -1)  # [B,1,G,G]

        prior_map = torch.cat([prior_space, prior_border], dim=1)  # [B,2,G,G]
        logits_prior = self.prior_fuse(prior_map)  # [B,1,G,G]

        wire_abs = wire.abs()
        wire_feat = torch.stack([
            wire_abs.mean(dim=[1, 2, 3]),  # [B]
            wire_abs.amax(dim=[1, 2, 3])  # [B]
        ], dim=1)  # [B,2]
        gate = self.gate_mlp(wire_feat).view(B, 1, 1, 1)  # [B,1,1,1] in (0,1)

        logits = gate * logits_data + (1.0 - gate) * logits_prior  # [B,1,G,G]

        invalid = (posmsk >= 1.0)  # [B,1,G,G]
        logits = logits.masked_fill(invalid, -1.0e10)

        logits = logits.view(B, -1)  # [B, G*G]
        probs = self.softmax(logits / self.tau.clamp_min(1e-3))

        return probs


# ========== Critic ==========
class RDC2(nn.Module):
    def __init__(self, input_channels=3, grid=224, hidden_dim=256, lstm_hidden_dim=64):
        super().__init__()
        self.image_size = grid

        # Shared CNN feature extractor
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.cnn_output_dim = 64 * 4 * 4

        # Shared FC layer
        self.shared_fc = nn.Sequential(
            nn.Linear(self.cnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # total value
        self.value_total_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # imm value
        self.value_imm_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.value_imm_fc = nn.Linear(64, 1)

        # future value
        self.value_future_lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        self.value_future_fc = nn.Linear(lstm_hidden_dim, 1)

        self.early_features = None

    def hook_fn(self, module, input, output):
        self.early_features = output

    def forward(self, x, sequence_length=1):
        """
        Args:
            x: input [batch_size, channels, height, width]
            sequence_length: default
        """
        batch_size = x.size(0)

        hook_handle = None
        if hasattr(self.shared_cnn[0], 'register_forward_hook'):
            hook_handle = self.shared_cnn[2].register_forward_hook(self.hook_fn)

        cnn_features = self.shared_cnn(x)
        cnn_features_flat = cnn_features.view(batch_size, -1)

        shared_features = self.shared_fc(cnn_features_flat)

        v_total = self.value_total_head(shared_features)

        v_imm = torch.zeros(batch_size, 1, device=x.device)
        if self.early_features is not None:
            imm_features = self.value_imm_conv(self.early_features)
            imm_features_flat = imm_features.view(batch_size, -1)
            v_imm = self.value_imm_fc(imm_features_flat)

        if hook_handle is not None:
            hook_handle.remove()
        self.early_features = None

        shared_features_seq = shared_features.unsqueeze(1)  # [batch, 1, features]
        if sequence_length > 1:
            shared_features_seq = shared_features_seq.repeat(1, sequence_length, 1)
        lstm_out, _ = self.value_future_lstm(shared_features_seq)
        v_future = self.value_future_fc(lstm_out[:, -1, :])  # take last output

        return v_total, v_imm, v_future


# ========== PPO ==========
class PPO:
    def __init__(self, env, args):
        self.env = env
        self.device = args.device
        self.gcn = None
        self.actor_net = Actor(grid=args.grid).float().to(self.device )
        self.critic_net = RDC2(grid=args.grid).float().to(self.device )

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.A_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), args.C_lr)

        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.ppo_epoch = args.ppo_epoch
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.args = args
        self.placed_num_macro = args.pnm
        self.buffer_capacity = args.buffer_capacity * args.pnm

    def select_action(self, state, Eval=False):
        state = state.clone().detach().to(self.device).float()

        state = state.unsqueeze(0)
        with torch.no_grad():
            probs = self.actor_net(state)

        dist = Categorical(probs)

        if Eval:
            action = torch.argmin(probs, dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob

    def store_transition(self, transition):
        # Helper function to convert data to tensor if it's not already in tensor format
        def to_tensor(data, dtype=torch.float):
            if isinstance(data, torch.Tensor):
                return data
            return torch.tensor(data, dtype=dtype, device=self.device)

        # Standardizing transition components
        state = to_tensor(transition.state, dtype=torch.float)
        next_state = to_tensor(transition.next_state, dtype=torch.float)
        reward = to_tensor(transition.reward, dtype=torch.float)
        a_log_prob = to_tensor(transition.a_log_prob, dtype=torch.float)
        action = to_tensor(transition.action, dtype=torch.long)
        # Append the transition to the buffer
        self.buffer.append(Transition(state, action, reward, a_log_prob, next_state))

        # Increment the counter and check if buffer capacity is reached
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self, writer=None):
        states = torch.stack([t.state for t in self.buffer]).to(self.device)
        actions = torch.stack([t.action for t in self.buffer]).view(-1, 1).to(self.device)
        rewards = torch.stack([t.reward for t in self.buffer]).view(-1, 1).to(self.device)
        old_log_probs = torch.stack([t.a_log_prob for t in self.buffer]).view(-1, 1).to(self.device)

        target_list = []
        target = 0
        for i in reversed(range(rewards.shape[0])):
            if self.env.t >= self.placed_num_macro - 1:
                target = 0
            target = rewards[i, 0].item() + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_v_all = torch.tensor(target_list, dtype=torch.float, device=self.device).view(-1, 1)

        self.buffer.clear()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True):
                self.training_step += 1

                probs = self.actor_net(states[index].to(self.device))
                dist = Categorical(probs)
                action_log_prob = dist.log_prob(actions[index].squeeze())
                ratio = torch.exp(action_log_prob - old_log_probs[index].squeeze())

                # ------- Critic with Return Decomposition -------
                v_total, v_imm, v_future = self.critic_net(states[index].to(self.device))

                # advantage = target - total
                advantage = (target_v_all[index] - v_total).detach()

                # Actor loss
                surr1 = ratio * advantage.squeeze()
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage.squeeze()
                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic loss
                value_loss_total = F.smooth_l1_loss(v_total, target_v_all[index])
                value_loss_balance = F.mse_loss(v_total, v_imm + v_future)
                value_loss = value_loss_total + 0.1 * value_loss_balance

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                if writer:
                    writer.add_scalar('ppo/action_loss', actor_loss.item(), self.training_step)
                    writer.add_scalar('ppo/value_loss_total', value_loss_total.item(), self.training_step)
                    writer.add_scalar('ppo/value_loss_balance', value_loss_balance.item(), self.training_step)

    def save_param(self, path):
        torch.save({
            'actor_net_dict': self.actor_net.state_dict(),
            'critic_net_dict': self.critic_net.state_dict()
        }, path)

    def load_param(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_net.load_state_dict(checkpoint['actor_net_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_net_dict'])
