# By Joseph Trevorrow (University Of Liverpool)
# Modified MADDPG algorithm using a vdn style value function factorisation method

import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.maddpg import MADDPGCritic
import torch as th
from torch.optim import RMSprop, Adam
from controllers.maddpg_controller import gumbel_softmax
from modules.critics import REGISTRY as critic_registry
from components.standarize_stream import RunningMeanStd

from modules.mixers.nmaddpg import nmadddpg



class MADDPG_NonMon_Learner:
    def __init__(self, mac, scheme, logger, args):
        # Create our parameters
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        self.agent_optimiser = Adam(params=self.agent_params, lr=self.args.lr)
        self.critic_optimiser = Adam(params=self.critic_params, lr=self.args.lr)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.last_target_update_episode = 0

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)
        
        # Mixer
        self.mixer = nmadddpg(args)
        self.target_mixer = copy.deepcopy(self.mixer)

    # The backpropagation step
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities from a batch
        OriRewards = batch["reward"][:, :-1]
        actions = batch["actions_onehot"]
        OriTerminated = batch["terminated"][:, :-1].float()
        rewards = OriRewards.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        terminated = OriTerminated.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        
        #mask = 1 - terminated

        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - OriTerminated[:, :-1])
        
        
        batch_size = batch.batch_size

        # Do some normalisation if necessary
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Train the critic; This has to be first, as the actor uses the Q value taken from the critic
        inputs = self._build_inputs(batch)
        actions = actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)
        critic_out = self.critic(inputs[:, :-1], actions[:, :-1].detach())
        # Have our Q Value, this is what the critic will hold, and therefore the actor will use to update also
        q_taken = critic_out.view(batch_size, -1, 1)

        # Compute a forward propagation step on this batch, to get target values (agent output)
        self.target_mac.init_hidden(batch.batch_size)
        target_actions = []
        for t in range(1, batch.max_seq_length):
            # Target actions is the actions taken from 1 forward prop step (the agents output)
            agent_target_outs = self.target_mac.target_actions(batch, t)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)  # Concat over time

        # run forward prop of the actions the actor took, through the critic (some relu, etc.)
        target_actions = target_actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)
        target_vals = self.target_critic(inputs[:, 1:], target_actions.detach())
        target_vals = target_vals.view(batch_size, -1, 1)

        # Mix here our target q values from the forward prop
        if self.mixer is not None:
            mixed_q_taken = self.mixer(q_taken, batch["state"][:, :-1]).view(batch_size, -1, 1)
            mixed_target_vals = self.target_mixer(target_vals,  batch["state"][:, 1:])

        # Normalise
        if self.args.standardise_returns:
            mixed_target_vals = mixed_target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = OriRewards.reshape(-1, 1) + self.args.gamma * (1 - OriTerminated.reshape(-1, 1)) * mixed_target_vals.reshape(-1, 1).detach()
        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # The Critic's Q value is being used here to get the TD error and loss
        td_error = (mixed_q_taken.view(-1, 1) - targets.detach())
        masked_td_error = td_error * mask.reshape(-1, 1)
        loss = (masked_td_error ** 2).mean()

        # Backpropagate over the critic, to modify the q value taken (This is where the Q value then is being changed)
        self.critic_optimiser.zero_grad()
        loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        # END OF CRITIC TRAINING, ACTOR BELOW:

        # Train the actor using the sampled policy gradient from the mini batch before
        self.mac.init_hidden(batch_size)
        pis = []
        actions = []
        for t in range(batch.max_seq_length-1):
            pi = self.mac.forward(batch, t=t).view(batch_size, 1, self.n_agents, -1)
            actions.append(gumbel_softmax(pi, hard=True))
            pis.append(pi)
        actions = th.cat(actions, dim=1)
        actions = actions.view(batch_size, -1, 1, self.n_agents * self.n_actions).expand(-1, -1, self.n_agents, -1)

        new_actions = []
        for i in range(self.n_agents):
            temp_action = th.split(actions[:, :, i, :], self.n_actions, dim=2)
            actions_i = []
            for j in range(self.n_agents):
                if i == j:
                    actions_i.append(temp_action[j])
                else:
                    actions_i.append(temp_action[j].detach())
            actions_i = th.cat(actions_i, dim=-1)
            new_actions.append(actions_i.unsqueeze(2))
        new_actions = th.cat(new_actions, dim=2)

        pis = th.cat(pis, dim=1)
        pis[pis==-1e10] = 0
        pis = pis.reshape(-1, 1)

        # Here we collect every q value
        q = self.critic(inputs[:, :-1], new_actions)
        q = q.reshape(-1, 1)
        mask = mask.reshape(-1, 1)


        # Compute the actor loss with respect to every q and policy pi
        pg_loss = -(q * mask.reshape(1, -1)).mean() + self.args.reg * (pis ** 2).mean()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if self.args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_taken_mean", (mixed_q_taken).sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", mixed_target_vals.sum().item() / mask_elems, t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            self.log_stats_t = t_env

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = []
        inputs.append(batch["state"][:, ts].unsqueeze(2).expand(-1, -1, self.n_agents, -1))
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts])

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t - 1, t)])
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]],
                                      dim=1)
                # last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        # Update the mixer
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        # Update the mixer
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
        # Load in mixer to cuda
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        # Save the mixer
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        # Load in the mixer
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))