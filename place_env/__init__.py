from gym.envs.registration import register

register(
    id='place_env-v0',
    entry_point='place_env.place_env:PlaceEnv'
)

register(
    id='place_env-gpu-v1',
    entry_point='place_env.place_env_gpu:PlaceEnvGpu'
)

register(
    id='place_env-gpu-v2',
    entry_point='place_env.place_env_gpu_v2:PlaceEnvGpu'
)

register(
    id='place_env-for-long-v1',
    entry_point='place_env.place_env_for_long:PlaceEnvGpu'
)

register(
    id='fast_place-v0',
    entry_point='place_env.fast_env:PlaceEnvGpu'
)

register(
    id='fast_place-v1',
    entry_point='place_env.fast_env_gpu_wRUDY:PlaceEnvGpu'
)

register(
    id='greedy_rollout-v1',
    entry_point='place_env.greedy_rollout:PlaceEnvGpu'
)
