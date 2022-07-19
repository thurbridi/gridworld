from gym.envs.registration import register

register(
    id="grid_world/FourRooms-v0",
    entry_point="grid_world.envs:FourRoomsEnv",
    max_episode_steps=300,
)
