"""MuJoCo model loading and RobotWrapper: Jacobians, pose, inertia, gravity, torque."""

import logging
from pathlib import Path

import numpy as np
import mujoco

logger = logging.getLogger(__name__)


def load_model(model_xml_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a MuJoCo model from XML and create its data buffer.

    Args:
        model_xml_path: Path to the model XML file.

    Returns:
        Tuple of (MjModel, MjData).
    """
    model = mujoco.MjModel.from_xml_path(str(model_xml_path))
    data = mujoco.MjData(model)
    return model, data


class RobotWrapper:
    """Wrapper around a MuJoCo model for end-effector and dynamics queries.

    Caches end-effector site/body IDs and provides Jacobians, pose, inertia,
    gravity, Coriolis/centrifugal, and torque get/set. Uses a scratch MjData
    for gravity computation to avoid per-call allocation.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str | None = None,
        body_name: str | None = None,
    ) -> None:
        """Build wrapper; at least one of site_name or body_name must be given for EE methods.

        Args:
            model: MuJoCo model.
            site_name: Name of end-effector site (optional).
            body_name: Name of end-effector body (optional).

        Raises:
            ValueError: If a given site_name or body_name is not found in the model.
        """
        self.model = model
        self.ee_site_id = -1
        self.ee_body_id = -1
        if site_name is not None:
            self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if self.ee_site_id < 0:
                logger.warning("site_name '%s' not found in model", site_name)
                raise ValueError(f"site_name '{site_name}' not found")
        if body_name is not None:
            self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if self.ee_body_id < 0:
                logger.warning("body_name '%s' not found in model", body_name)
                raise ValueError(f"body_name '{body_name}' not found")
        self._scratch = mujoco.MjData(model)

    def _jacobian(
        self,
        data: mujoco.MjData,
        use_site: bool,
        obj_id: int,
        translational: bool,
        rotational: bool,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        return _jacobian_impl(
            self.model, data, use_site, obj_id, translational, rotational
        )

    def get_ee_jacobian(
        self,
        data: mujoco.MjData,
        translational: bool = True,
        rotational: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Jacobian for the configured end-effector (site or body). Linear vel v = Jp @ qd, angular vel omega = Jr @ qd.

        Args:
            data: Simulation state.
            translational: Include translational part Jp.
            rotational: Include rotational part Jr.

        Returns:
            (J, Jp, Jr): J = stacked [Jp; Jr] when both requested (6 x nv), else 3 x nv.
            Jp is None when rotational-only; Jr is None when translational-only.
            Callers can use J, or Jp, or Jr as needed.

        Raises:
            ValueError: If neither site_name nor body_name was set at construction.
        """
        if self.ee_site_id >= 0:
            raw = self._jacobian(data, True, self.ee_site_id, translational, rotational)
        elif self.ee_body_id >= 0:
            raw = self._jacobian(data, False, self.ee_body_id, translational, rotational)
        else:
            raise ValueError("EE not configured. Pass site_name or body_name to RobotWrapper.")
        return _jacobian_to_J_Jp_Jr(raw, translational, rotational)

    def get_site_jacobian(
        self,
        data: mujoco.MjData,
        site_name: str,
        translational: bool = True,
        rotational: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Jacobian for a site by name. Returns (J, Jp, Jr) as in get_ee_jacobian."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            logger.warning("site_name '%s' not found in model", site_name)
            raise ValueError(f"site_name '{site_name}' not found")
        raw = _jacobian_impl(
            self.model, data, True, site_id, translational, rotational
        )
        return _jacobian_to_J_Jp_Jr(raw, translational, rotational)

    def get_body_jacobian(
        self,
        data: mujoco.MjData,
        body_name: str,
        translational: bool = True,
        rotational: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Jacobian for a body COM by name. Returns (J, Jp, Jr) as in get_ee_jacobian."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            logger.warning("body_name '%s' not found in model", body_name)
            raise ValueError(f"body_name '{body_name}' not found")
        raw = _jacobian_impl(
            self.model, data, False, body_id, translational, rotational
        )
        return _jacobian_to_J_Jp_Jr(raw, translational, rotational)

    def get_ee_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Position and orientation of the configured end-effector.

        Returns:
            (pos, R): position vector (3,) and rotation matrix (3,3).

        Raises:
            ValueError: If EE not configured.
        """
        if self.ee_site_id >= 0:
            return _get_pose_impl(self.model, data, True, self.ee_site_id)
        if self.ee_body_id >= 0:
            return _get_pose_impl(self.model, data, False, self.ee_body_id)
        raise ValueError("EE not configured. Pass site_name or body_name to RobotWrapper.")

    def get_pose(
        self,
        data: mujoco.MjData,
        site_name: str | None = None,
        body_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Position and orientation of a site or body by name.

        Args:
            data: Simulation state.
            site_name: Name of the site (optional).
            body_name: Name of the body (optional).

        Returns:
            (pos, R): position (3,) and rotation matrix (3,3).

        Raises:
            ValueError: If neither name given or name not found.
        """
        if site_name is not None:
            site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            if site_id < 0:
                logger.warning("site_name '%s' not found in model", site_name)
                raise ValueError(f"site_name '{site_name}' not found")
            return _get_pose_impl(self.model, data, True, site_id)
        if body_name is not None:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                logger.warning("body_name '%s' not found in model", body_name)
                raise ValueError(f"body_name '{body_name}' not found")
            return _get_pose_impl(self.model, data, False, body_id)
        raise ValueError("One of site_name or body_name must be specified.")

    def get_gravity_vector(self, data: mujoco.MjData) -> np.ndarray:
        """Generalized gravity vector g(q). Inverse dynamics with qd=0, qdd=0 gives qfrc_inverse. Uses pre-allocated scratch MjData.

        Args:
            data: Current state (qpos used).

        Returns:
            Gravity vector (nv,) in joint space.
        """
        self._scratch.qpos[:] = data.qpos
        mujoco.mj_inverse(self.model, self._scratch)
        return np.copy(self._scratch.qfrc_inverse)

    def get_coriolis_centrifugal(self, data: mujoco.MjData) -> np.ndarray:
        """Coriolis and centrifugal term C(q,qd)*qd. Bias b = C*qd + g(q), so C*qd = b - g.

        Args:
            data: Current state.

        Returns:
            Vector (nv,) equal to C(q,qd)*qd.
        """
        g = self.get_gravity_vector(data)
        return np.copy(data.qfrc_bias) - g

    def get_inertia_matrix(self, data: mujoco.MjData) -> np.ndarray:
        """Inertia matrix M(q) (mass matrix). Equation of motion: M(q)*qdd + C(q,qd)*qd + g(q) = tau.

        Args:
            data: Current state.

        Returns:
            Symmetric positive definite matrix (nv, nv).
        """
        nv = self.model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(self.model, M, data.qM)
        return M

    def get_bias_forces(self, data: mujoco.MjData) -> np.ndarray:
        """Bias forces b = C(q,qd)*qd + g(q).

        Args:
            data: Current state.

        Returns:
            Vector (nv,).
        """
        return np.copy(data.qfrc_bias)

    def get_torque(self, data: mujoco.MjData) -> np.ndarray:
        """Current actuator torque (control input) from data.ctrl.

        Returns:
            Copy of data.ctrl (nu,).
        """
        return np.copy(data.ctrl)

    def set_torque(self, data: mujoco.MjData, tau: np.ndarray) -> None:
        """Set actuator torque (control input).

        Args:
            data: Simulation data to modify.
            tau: Torque vector; truncated or zero-padded to match data.ctrl length.
        """
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        nu = len(data.ctrl)
        n = min(len(tau), nu)
        data.ctrl[:n] = tau[:n]
        if n < nu:
            data.ctrl[n:] = 0.0


def _jacobian_to_J_Jp_Jr(
    raw: np.ndarray | tuple[np.ndarray, np.ndarray],
    translational: bool,
    rotational: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Convert _jacobian_impl return value to (J, Jp, Jr). J = [Jp; Jr] when both, else the single part."""
    if isinstance(raw, tuple):
        Jp, Jr = raw
        return (np.vstack([Jp, Jr]), Jp, Jr)
    if translational:
        return (raw, raw, None)
    return (raw, None, raw)


def _jacobian_impl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    use_site: bool,
    obj_id: int,
    translational: bool,
    rotational: bool,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute Jacobian for a site or body COM: v = Jp @ qd, omega = Jr @ qd."""
    nv = model.nv
    jacp = np.zeros((3, nv)) if translational else None
    jacr = np.zeros((3, nv)) if rotational else None
    if use_site:
        mujoco.mj_jacSite(model, data, jacp, jacr, obj_id)
    else:
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, obj_id)
    if translational and rotational:
        return jacp, jacr
    return jacp if translational else jacr


def _get_pose_impl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    use_site: bool,
    obj_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Position p and rotation matrix R (3x3) for a site or body."""
    if use_site:
        pos = np.copy(data.site_xpos[obj_id])
        mat = np.copy(data.site_xmat[obj_id]).reshape(3, 3)
    else:
        pos = np.copy(data.xpos[obj_id])
        mat = np.copy(data.xmat[obj_id]).reshape(3, 3)
    return pos, mat
