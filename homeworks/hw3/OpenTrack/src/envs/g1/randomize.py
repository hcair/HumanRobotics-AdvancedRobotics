# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for randomization."""
import jax
from mujoco import mjx


TORSO_BODY_ID = 16


def domain_randomize_model(model: mjx.Model, rng: jax.Array):
    @jax.vmap
    def rand_dynamics(rng):
        # Floor / foot friction: =U(0.3, 2.0).
        rng, key = jax.random.split(rng)
        friction = jax.random.uniform(key, minval=0.3, maxval=2.0)
        # left_shin_floor, left_foot_floor, right_shin_floor, right_foot_floor, torso_floor, left_hand_floor, right_hand_floor
        pair_friction = model.pair_friction.at[1, 0:2].set(friction)
        pair_friction = model.pair_friction.at[3, 0:2].set(friction)

        # Scale static friction: *U(0.5, 2.0).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(key, shape=(29,), minval=0.5, maxval=2.0)
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[6:] * jax.random.uniform(key, shape=(29,), minval=1.0, maxval=1.05)
        dof_armature = model.dof_armature.at[6:].set(armature)

        # Jitter center of mass positiion: +U(-0.15, 0.15).
        rng, key = jax.random.split(rng)
        dpos = jax.random.uniform(key, (3,), minval=-0.15, maxval=0.15)
        body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(model.body_ipos[TORSO_BODY_ID] + dpos)

        # Add mass to torso: +U(-3.0, 6.0).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=-3.0, maxval=6.0)
        ratio = dmass / model.body_mass[TORSO_BODY_ID]
        body_mass = model.body_mass.at[TORSO_BODY_ID].set(model.body_mass[TORSO_BODY_ID] * (1 + ratio))
        body_inertia = model.body_inertia.at[TORSO_BODY_ID].set(model.body_inertia[TORSO_BODY_ID] * (1 + ratio)[..., None])

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(qpos0[7:] + jax.random.uniform(key, shape=(29,), minval=-0.05, maxval=0.05))

        return (
            pair_friction,
            dof_frictionloss,
            dof_armature,
            body_ipos,
            body_mass,
            body_inertia,
            qpos0,
        )

    (
        pair_friction,
        frictionloss,
        armature,
        body_ipos,
        body_mass,
        body_inertia,
        qpos0,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
        {
            "pair_friction": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "body_ipos": 0,
            "body_mass": 0,
            "body_inertia": 0,
            "qpos0": 0,
        }
    )

    model = model.tree_replace(
        {
            "pair_friction": pair_friction,
            "dof_frictionloss": frictionloss,
            "dof_armature": armature,
            "body_ipos": body_ipos,
            "body_mass": body_mass,
            "body_inertia": body_inertia,
            "qpos0": qpos0,
        }
    )

    return model, in_axes


def domain_randomize_terrain(model: mjx.Model, rng: jax.Array, all_hfield_data: jax.Array):
    @jax.vmap
    def rand_dynamics(rng):
        # Floor / foot friction: =U(0.3, 2.0).
        rng, key = jax.random.split(rng)
        friction = jax.random.uniform(key, minval=0.3, maxval=2.0)
        # left_shin_floor, left_foot_floor, right_shin_floor, right_foot_floor, torso_floor, left_hand_floor, right_hand_floor
        pair_friction = model.pair_friction.at[1, 0:2].set(friction)
        pair_friction = model.pair_friction.at[3, 0:2].set(friction)

        # Scale static friction: *U(0.5, 2.0).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(key, shape=(29,), minval=0.5, maxval=2.0)
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[6:] * jax.random.uniform(key, shape=(29,), minval=1.0, maxval=1.05)
        dof_armature = model.dof_armature.at[6:].set(armature)

        # Jitter center of mass positiion: +U(-0.15, 0.15).
        rng, key = jax.random.split(rng)
        dpos = jax.random.uniform(key, (3,), minval=-0.15, maxval=0.15)
        body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(model.body_ipos[TORSO_BODY_ID] + dpos)

        # Add mass to torso: +U(-3.0, 6.0).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=-3.0, maxval=6.0)
        ratio = dmass / model.body_mass[TORSO_BODY_ID]
        body_mass = model.body_mass.at[TORSO_BODY_ID].set(model.body_mass[TORSO_BODY_ID] * (1 + ratio))
        body_inertia = model.body_inertia.at[TORSO_BODY_ID].set(model.body_inertia[TORSO_BODY_ID] * (1 + ratio)[..., None])

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(qpos0[7:] + jax.random.uniform(key, shape=(29,), minval=-0.05, maxval=0.05))

        # Randomize terrain: scale (10.0, 16.0), octaves (5, 8), persistence (0.3, 0.5), lacunarity (2.0, 4.0)
        rng, key = jax.random.split(rng)
        hfield_data = all_hfield_data[jax.random.randint(key, shape=(), minval=0, maxval=1024)]

        return (
            pair_friction,
            dof_frictionloss,
            dof_armature,
            body_ipos,
            body_mass,
            body_inertia,
            qpos0,
            hfield_data,
        )

    (
        pair_friction,
        frictionloss,
        armature,
        body_ipos,
        body_mass,
        body_inertia,
        qpos0,
        hfield_data,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
        {
            "pair_friction": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "body_ipos": 0,
            "body_mass": 0,
            "body_inertia": 0,
            "qpos0": 0,
            "hfield_data": 0,
        }
    )

    model = model.tree_replace(
        {
            "pair_friction": pair_friction,
            "dof_frictionloss": frictionloss,
            "dof_armature": armature,
            "body_ipos": body_ipos,
            "body_mass": body_mass,
            "body_inertia": body_inertia,
            "qpos0": qpos0,
            "hfield_data": hfield_data,
        }
    )

    return model, in_axes
