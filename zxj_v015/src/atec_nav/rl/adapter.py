import numpy as np
from gymnasium import spaces


class DreamerAdapter:
    """Duck-typed embodied.Env for flat Gymnasium Dict observations.

    The upstream FromGym adapter expects the legacy four-value step API.
    Import elements lazily so contract tests do not require JAX or Dreamer.
    """

    def __init__(self, env, seed=0):
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("Expected flat Dict observations")
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("Expected continuous Box actions")
        if any(not isinstance(s, spaces.Box)
               for s in env.observation_space.spaces.values()):
            raise TypeError("Observation entries must be Box spaces")
        reserved = {"reward", "is_first", "is_last", "is_terminal"}
        if reserved.intersection(env.observation_space.spaces):
            raise ValueError("Observation keys collide with Dreamer metadata")
        self.env = env
        self.seed = seed
        self.done = True
        self.info = {}

    @property
    def obs_space(self):
        import elements

        result = {
            key: elements.Space(space.dtype, space.shape, space.low, space.high)
            for key, space in self.env.observation_space.spaces.items()
        }
        result.update(
            reward=elements.Space(np.float32),
            is_first=elements.Space(bool),
            is_last=elements.Space(bool),
            is_terminal=elements.Space(bool),
        )
        return result

    @property
    def act_space(self):
        import elements

        space = self.env.action_space
        return {
            "action": elements.Space(
                space.dtype, space.shape, space.low, space.high),
            "reset": elements.Space(bool),
        }

    def step(self, action):
        if bool(action["reset"]) or self.done:
            observation, self.info = self.env.reset(seed=self.seed)
            self.seed = None
            self.done = False
            return self._pack(observation, 0.0, True, False, False)
        observation, reward, terminated, truncated, self.info = self.env.step(
            np.asarray(action["action"], dtype=self.env.action_space.dtype))
        self.done = bool(terminated or truncated)
        return self._pack(
            observation, reward, False, self.done, bool(terminated))

    @staticmethod
    def _pack(observation, reward, first, last, terminal):
        return {
            **{k: np.asarray(v) for k, v in observation.items()},
            "reward": np.float32(reward),
            "is_first": np.bool_(first),
            "is_last": np.bool_(last),
            "is_terminal": np.bool_(terminal),
        }

    def close(self):
        self.env.close()

