from environment.snake.snake_env import SnakeEnv
from agent.dqn import DQNAgent, Resize, ToTensor
import torch
import torchvision.transforms as transforms


if __name__ == "__main__":
    n_games = 100000
    screen_size = (80, 80)
    frame_count = 0
    UPDATE_TARGET_FRAME = 2000
    START_LEARNING_FRAME = 500

    dqn_agent = DQNAgent(
        input_shape=(4, *screen_size),
        gamma=0.99,
        memory_capacity=1000000,
        epsilon=1,
        epsilon_decay=0.999,
        min_epsilon=0.01,
        batch_size=32,
        lr=0.00025,
        transform=transforms.Compose([ToTensor()])
    )
    env = SnakeEnv(screen_size)
    scores = []
    max_score = 0
    # TODO Create a monitoring class to monitor reward scores?
    for episode in range(n_games):
        observation = env.get_observation()
        while not env.done:
           
            state = dqn_agent.process(observation)
            action = dqn_agent.choose_next_action(state=state)
            (reward, next_observation, is_done) = env.step(action)
           
            
            dqn_agent.store_transition(
                observation, action, reward, next_observation, is_done
            )
            if frame_count > START_LEARNING_FRAME:
                dqn_agent.learn()
            observation = next_observation
            
            
            frame_count += 1
            # TODO Agent should be aware of how many frames are processed to do this updates by himself
            if frame_count % UPDATE_TARGET_FRAME == 0:
                dqn_agent.update_target_network()
        if frame_count > START_LEARNING_FRAME:
            dqn_agent.update_epsilon()
        if env.score > max_score:
            max_score = env.score
            print(f"New max score {max_score}")
        env.reset()
        
        scores.append(env.score)
    dqn_agent.save_model("model.pt")
