import gym
from gym import spaces
import numpy as np
from enum import IntEnum


class FourRoomsEnv(gym.Env):
    """Unofficial implementation of the GridWorld environment used in "Discovery of Options via Meta-Learned Subgoals" paper."""

    Movement = IntEnum("Movement", [("UP", 0), ("DOWN", 1), ("LEFT", 2), ("RIGHT", 3)])
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.layout, self.target_regions = self._make_gridworld()

        self.agent_pos = None
        self.target_pos = None

        self._is_reset = False

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, *self.layout.shape), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            self.Movement.UP: np.array([-1, 0]),
            self.Movement.DOWN: np.array([1, 0]),
            self.Movement.LEFT: np.array([0, -1]),
            self.Movement.RIGHT: np.array([0, 1]),
        }

        if self.render_mode == "human":
            self.window_size = 512

            import pygame  # import here to avoid pygame dependency with no render

            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def step(self, action):
        assert self._is_reset, "Cannot call env.step() before calling reset()"
        self.agent_pos = self._move(action)

        obs = self._get_obs()

        reward = 1 if self._is_agent_on_target() else 0

        done = self._is_agent_on_target()

        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, done, info

    def reset(self):
        self._is_reset = True
        self.target_pos = self._sample_target_pos()
        self.agent_pos = self._sample_agent_pos()

        obs = self._get_obs()

        return obs

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

    def _render_frame(self):
        import pygame  # avoid global pygame dependency. This method is not called with no-render.

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.layout.shape[0]
        )  # The size of a single grid square in pixels

        for idx, val in np.ndenumerate(self.layout):
            if val == 1:
                pygame.draw.rect(
                    canvas,
                    (0, 0, 0),
                    pygame.Rect(
                        pix_square_size * np.array(idx[::-1]),
                        (pix_square_size, pix_square_size),
                    ),
                )

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.target_pos[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.agent_pos[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.layout.shape[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            assert self.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array or single_rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _sample_target_pos(self):
        target_regions = self.target_regions["train"]
        sampling_region = target_regions[np.random.randint(target_regions.shape[0])]
        target_pos = np.array(
            [
                np.random.randint(sampling_region[0, 0], sampling_region[1, 0]),
                np.random.randint(sampling_region[0, 1], sampling_region[1, 1]),
            ]
        )

        return target_pos

    def _sample_agent_pos(self):
        agent_pos = np.random.randint(1, self.layout.shape[1] - 1, size=2)
        while (
            np.array_equal(agent_pos, self.target_pos)
            or self.layout[agent_pos[0], agent_pos[1]] == 1
        ):
            agent_pos = np.random.randint(1, self.layout.shape[1] - 1, size=2)

        return agent_pos

    def _make_gridworld(self):
        layout = self._make_layout()
        target_regions = self._make_target_regions(layout)
        return layout, target_regions

    def _make_layout(self):
        layout = np.zeros((13, 13), dtype=np.uint8)
        # outter walls
        layout[0, :] = 1
        layout[-1, :] = 1
        layout[:, 0] = 1
        layout[:, -1] = 1

        # inner walls
        layout[:, 6] = 1
        layout[6, :6] = 1
        layout[7, 6:] = 1

        # hallways
        pos_hallways = [(3, 6), (-3, 6), (6, 2), (7, -4)]
        for x, y in pos_hallways:
            layout[x, y] = 0

        return layout

    def _make_target_regions(self, layout):
        target_regions = {
            "train": np.array(
                [
                    [[2, 2], [5, 5]],
                    [[2, 8], [5, 11]],
                    [[9, 8], [12, 11]],
                ]
            ),
            "test": np.array([[[8, 2], [11, 5]]]),
        }

        # TODO: assert targets are in shape limits

        return target_regions

    def _move(self, action):
        new_agent_pos = self.agent_pos + self._action_to_direction[action]

        if self.layout[new_agent_pos[0], new_agent_pos[1]] == 1:
            return self.agent_pos

        return new_agent_pos

    def _is_agent_on_target(self):
        return np.array_equal(self.agent_pos, self.target_pos)

    def _get_obs(self):
        """Returns the grid layout in one channel, the agent\'s position in another channel,
        and the target positions in a third channel during training"""
        agent_channel = np.zeros_like(self.layout)
        agent_channel[self.agent_pos[0], self.agent_pos[1]] = 1

        target_channel = np.zeros_like(self.layout)
        target_channel[self.target_pos[0], self.target_pos[1]] = 1

        obs = np.stack([self.layout, agent_channel, target_channel], axis=0)

        return obs

    def _get_info(self):
        return dict()


if __name__ == "__main__":
    from gym.utils.env_checker import check_env

    check_env(FourRoomsEnv())

    env = FourRoomsEnv(render_mode="human")

    obs = env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break

    env.close()
