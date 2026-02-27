"""Focused tests for MultiEpisodeEnv integrations."""

from __future__ import annotations

import pytest

from envs.multi_episode_env import MultiEpisodeEnv


def test_frozenlake_adapter_episode_success_metadata() -> None:
    """MultiEpisodeEnv should detect success from FrozenLake adapter metadata."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    env = MultiEpisodeEnv(
        inner_env_class=FrozenLakeEnvAdapter,
        inner_env_kwargs={
            "env_id": "frozenlake",
            "env_kwargs": {
                "desc": ["SG", "FF"],
                "is_slippery": False,
                "max_turns": 3,
                "seed": 21,
            },
        },
        total_step_cap=3,
        success_reward=1.0,
        episode_header="New episode begins.",
        enable_reflection=False,
    )

    try:
        env.reset(seed=21, task={"seed": 21})
        _, reward, done, info = env.step(r"\boxed{right}")

        assert reward == 1.0
        assert done is False
        assert info["episode_done"] is True
        assert info["episode_success"] is True
        assert info["raw_reward"] == 1.0
        assert info["episode_index"] == 1
        assert info["episode_step"] == 0
    finally:
        env.close()


def test_frozenlake_reset_inner_env_reuses_same_task() -> None:
    """Internal reset should preserve the same FrozenLake task across episodes."""
    pytest.importorskip("gymnasium")
    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter

    def _snapshot(inner_adapter: FrozenLakeEnvAdapter) -> tuple[tuple[bytes, ...], int]:
        rows = tuple(b"".join(row.tolist()) for row in inner_adapter._env.desc)  # type: ignore[attr-defined]
        state = int(inner_adapter._env.s)  # type: ignore[attr-defined]
        return rows, state

    task = {
        "seed": 21,
        "desc": ["SG", "FF"],
        "is_slippery": False,
        "max_turns": 3,
    }
    env = MultiEpisodeEnv(
        inner_env_class=FrozenLakeEnvAdapter,
        inner_env_kwargs={
            "env_id": "frozenlake",
            "env_kwargs": {
                "desc": task["desc"],
                "is_slippery": task["is_slippery"],
                "max_turns": task["max_turns"],
                "seed": task["seed"],
            },
        },
        total_step_cap=4,
        success_reward=1.0,
        episode_header="New episode begins.",
        enable_reflection=False,
    )

    try:
        env.reset(seed=task["seed"], task=task)
        first_snapshot = _snapshot(env.inner_env)  # type: ignore[arg-type]

        # This terminal action triggers internal reset to episode 2.
        _, _, _, info = env.step(r"\boxed{right}")
        second_snapshot = _snapshot(env.inner_env)  # type: ignore[arg-type]

        assert info["episode_done"] is True
        assert info["episode_index"] == 1
        assert info["episode_step"] == 0
        assert first_snapshot == second_snapshot
    finally:
        env.close()
