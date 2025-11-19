
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from ActorCriticNetworks import ActorNetwork, CriticNetwork, copy_target, soft_update
from ReplayBuffer import ReplayBuffer
from helper import episode_reward_plot, video_agent
import numpy as np
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from gymnasium.wrappers import RecordVideo


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class TD3:
    """The TD3 Agent."""

    def __init__(self, env, replay_size=1000000, batch_size=32, gamma=0.99):
        """ Initializes the DQN method.
        
        Parameters
        ----------
        env: gym.Environment
            The gym environment the agent should learn in.
        replay_size: int
            The size of the replay buffer.
        batch_size: int
            The number of replay buffer entries an optimization step should be performed on.
        gamma: float
            The discount factor.      
        """

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma

        # Initialize Actor network and its target network.
        self.Actor = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        self.Actor_target = ActorNetwork(self.obs_dim, self.act_dim).to(device)

        # CHANGE 1: Initialize two Critic networks and their target networks.
        self.Critic1 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic1_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)

        self.Critic2 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic2_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)

        copy_target(self.Actor_target, self.Actor)
        copy_target(self.Critic1_target, self.Critic1)
        copy_target(self.Critic2_target, self.Critic2)


        # CHANGE 2: The actor network have a lower learning rate as proposed in the TD3 paper
        self.optim_actor = optim.Adam(self.Actor.parameters(), lr=0.001) 
        # CHANGE 3: The critic networks do not have weight decay in TD3
        self.optim_critic1 = optim.Adam(self.Critic1.parameters(), lr=0.001) 
        self.optim_critic2 = optim.Adam(self.Critic2.parameters(), lr=0.001)


    def learn(self, timesteps):
        """Train the agent for timesteps steps inside self.env.
        After every step taken inside the environment observations, rewards, etc. have to be saved inside the replay buffer.
        If there are enough elements already inside the replay buffer (>batch_size), compute MSBE loss and optimize DQN network.

        Parameters
        ----------
        timesteps: int
            Number of timesteps to optimize the DQN network.
        """
        all_rewards = []
        episode_rewards = []
        all_rewards_eval = []
        timeexit = timesteps

        # CHANGE 4: We use Normal Action Noise with mean 0 and std 0.1 as proposed in the TD3 paper
        NANoise =  NormalActionNoise(mean=np.zeros(self.act_dim), sigma=0.1*np.ones(self.act_dim))

        obs, _ = self.env.reset()
        for timestep in range(1, timesteps + 1):

            action = self.choose_action(obs)

            # Here we sample and add the noise to the action to explore the environment. Notice we clip the action
            # between -1 and 1 because the action space is continuous and bounded between -1 and 1.
            epsilon= OUNoise.sample()
            action = np.clip(action + epsilon, -1, 1)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, terminated, truncated)
            
            obs = next_obs
            episode_rewards.append(reward)
            
            if terminated or truncated:
                all_rewards_eval.append(self.eval_episodes())
                print('\rTimestep: ', timestep, '/' ,timesteps,' Episode reward: ',np.round(all_rewards_eval[-1]), 'Episode: ', len(all_rewards), 'Mean R', np.mean(all_rewards_eval[-100:]))
                obs, _ = self.env.reset()
                all_rewards.append(sum(episode_rewards))
                episode_rewards = []
                    
            if len(self.replay_buffer) > self.batch_size:
                #TODO (6): if there is enouygh data in the replay buffer, sample a batch and perform an optimization step
                # Batch is sampled from the replay buffer and containes a list of tuples (s, a, r, s', term, trunc)

                # Get the batch data
                state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, truncated_batch = self.replay_buffer.get(self.batch_size)
                # Compute the loss for the critic and update the critic network 
                critic_loss = self.compute_critic_loss((state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, truncated_batch))
                self.optim_critic.zero_grad()
                critic_loss.backward()
                self.optim_critic.step()

                # Compute the loss for the actor and update the actor network 
                actor_loss = self.compute_actor_loss((state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, truncated_batch))
                self.optim_actor.zero_grad()
                actor_loss.backward()
                self.optim_actor.step() 

                # END TODO (6)

            # The target networks have an update rate of 5*0.001 as proposed in the TD3 paper
            soft_update(self.Critic_target, self.Critic, tau=5*0.001)
            soft_update(self.Actor_target, self.Actor, tau=5^0.001)

            # END TODO (7)

            if timestep % (timesteps-1) == 0:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
                pass
            if len(all_rewards_eval)>10 and np.mean(all_rewards_eval[-5:]) > 220:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
                break
        return all_rewards, all_rewards_eval
    

    def choose_action(self, s):
        # TODO (3) Implement the function to choose an action given a state. It is deterministic because exploration is added
        # by the OrnsteinUhlenbeckActionNoise in the main loop.
        a = self.Actor(torch.tensor(s).to(device)).cpu().detach().numpy()
        # END TODO (3)
        return a


    def compute_critic_loss(self, batch):
        """
        The function computes the critic loss using the Mean Squared Bellman Error (MSBE) calculation.
        
        :param batch: The `batch` parameter is a tuple containing the data for computing the loss.
        :return: the critic loss, which is calculated using the mean squared error (MSE) loss between
        the expected Q-values (q_expected) and the target Q-values (target).
        """
        
        # TODO (4): Implement MSBE calculation (need to sample from replay buffer first). Notice that it is VERY 
        # similar to the DQN loss.

        # The batch is already sampled from the replay buffer previously (in TODO 6)
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, truncated_batch = batch
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
        terminated_batch = torch.FloatTensor(terminated_batch).to(device).unsqueeze(1)
        truncated_batch = torch.FloatTensor(truncated_batch).to(device).unsqueeze(1)

        q_targets_next = self.Critic_target(next_state_batch, self.Actor_target(next_state_batch))                                              
        target = reward_batch + (1-(terminated_batch)) *self.gamma*q_targets_next
        q_expected = self.Critic(state_batch, action_batch)

        criterion = nn.MSELoss()
        loss = criterion(q_expected, target)  

        # END TODO (4)
        return loss
    

    def compute_actor_loss(self,batch):
        """
        The function `compute_actor_loss` calculates the loss for the actor network 
        
        :param batch: The batch parameter is a tuple containing the data for computing the loss.
        :return: the loss, which is the negative mean of the expected Q-values.
        """
        # TODO (5) implement the actor loss. You have to sample from the replay buffer first a set of states.

        state_batch, _, _, _, _, _ = batch
        state_batch = torch.FloatTensor(state_batch).to(device)
 
        loss = -self.Critic(state_batch, self.Actor(state_batch)).mean()
        # END TODO (5) 

        return loss



    def eval_episodes(self,n=3):
        """ Evaluate an agent performing inside a Gym environment. """
        lr=[]
        for episode in range(n):
            tr = 0.0
            obs, _ = self.env.reset()
            while True:
                action = self.choose_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                tr += reward
                if terminated or truncated:
                    break
            lr.append(tr)
        return np.mean(lr)




if __name__ == '__main__':
    # Create gym environment
    env = gym.make("LunarLander-v3",continuous=True, render_mode='rgb_array')

    # We change the batch size to 100
    td3 = TD3(env,replay_size=1000000, batch_size=100, gamma=0.99)

    td3.learn(500000)
    env = RecordVideo(gym.make("LunarLander-v3",continuous=True, render_mode='rgb_array'),'video')    
    video_agent(env, td3,n_episodes=5)  
    pass
