import numpy as np
from collections import namedtuple, deque


class Memory():
    
    def __init__(self, action_size, buffer_size, batch_size):


        self.memory = deque(maxlen=buffer_size)  
        
        self.batch_size = batch_size

        self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


    def add_experience(state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience = self.experiences(state, action, reward, next_state, done)
        self.memory.append(experience)


    def sample_Batch():
        """Randomly sample a batch of experiences from memory."""
        experience_batch = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experience_batch if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experience_batch if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience_batch if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience_batch if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experience_batch if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    