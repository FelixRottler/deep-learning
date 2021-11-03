
from agent.dqn import DQNAgent, ToTensor
import torch
import torchvision.transforms as transforms
import gym

if __name__ == "__main__":
    n_games = 10000
    screen_size = (84, 84)
    gym_screen = (210,160)
    frame_count = 0
    UPDATE_TARGET_FRAME = 1500
    START_LEARNING_FRAME = 500

    dqn_agent = DQNAgent(
        memory_input_shape = (1,*gym_screen),
        input_shape=(1, 84, 84),
        gamma=0.99,
        memory_capacity=100000,
        epsilon=1,
        epsilon_decay=0.999,
        min_epsilon=0.01,
        batch_size=32,
        number_actions=4,
        lr=0.00025,
        transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((84,84)),transforms.Grayscale(1),transforms.ToTensor()]),
    )
    env = gym.make("Breakout-v0")
    
 
    scores = []
    max_score = 0
    # TODO Create a monitoring class to monitor reward scores?
    for episode in range(n_games):
        observation = env.reset().transpose(2,0,1)
        for t in range(100):
            env.render()
            state = dqn_agent.process(observation)
            action = dqn_agent.choose_next_action(state=state)
            next_observation,reward,is_done,_ =env.step(action) # take a random action
            next_observation = next_observation.transpose(2,0,1)
            dqn_agent.store_transition(
                observation, action, reward, next_observation, is_done
            )
            if frame_count > START_LEARNING_FRAME:
                dqn_agent.learn()
            observation = next_observation
            frame_count += 1
            if frame_count % UPDATE_TARGET_FRAME == 0:
                dqn_agent.update_target_network()
            if is_done:
                break
 
        if frame_count > START_LEARNING_FRAME:
            dqn_agent.update_epsilon()
        

       
    dqn_agent.save_model("model.pt")

