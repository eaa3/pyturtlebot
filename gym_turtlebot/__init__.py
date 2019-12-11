from gym.envs.registration import register

register(
    id='turtlebot-v0',
    entry_point='gym_turtlebot.envs.turtlebot_env:TurtleBotEnv',
)