# vla_framework/sim/mujoco_env.py
"""
MuJoCo Environment Wrapper
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)

class MuJoCoEnv:
    def __init__(self, xml_path="models/so101/scene.xml",
                 viewer=False, render_h=480, render_w=640,
                 timestep=None):
        import mujoco
        self._mj = mujoco
        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"MJCF not found: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data  = mujoco.MjData(self.model)
        if timestep is not None:
            self.model.opt.timestep = timestep
        self._render_h  = render_h
        self._render_w  = render_w
        self._renderer  = None
        self._viewer_on = viewer
        self._viewer    = None
        log.info("MuJoCoEnv loaded  nq=%d  nu=%d  dt=%.4f",
                 self.model.nq, self.model.nu, self.model.opt.timestep)

    def __enter__(self):
        self._renderer = self._mj.Renderer(
            self.model, height=self._render_h, width=self._render_w)
        if self._viewer_on:
            try:
                import mujoco.viewer as mjv
                self._viewer = mjv.launch_passive(self.model, self.data)
                log.info("Viewer launched")
            except Exception as e:
                log.warning("Viewer failed (%s) — headless", e)
                self._viewer = None
        return self

    def __exit__(self, *_):
        if self._renderer:
            self._renderer.close()
            self._renderer = None
        if self._viewer:
            try: self._viewer.close()
            except: pass
            self._viewer = None

    def step(self):
        self._mj.mj_step(self.model, self.data)

    def reset(self):
        self._mj.mj_resetData(self.model, self.data)
        self._mj.mj_forward(self.model, self.data)

    def forward(self):
        self._mj.mj_forward(self.model, self.data)

    def set_joint_positions(self, joint_names, values_rad):
        for name, val in zip(joint_names, values_rad):
            jid = self._mj.mj_name2id(
                self.model, self._mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint not found: {name!r}")
            self.data.qpos[self.model.jnt_qposadr[jid]] = float(val)
        self.forward()

    def get_joint_positions(self, joint_names):
        out = np.zeros(len(joint_names), dtype=np.float64)
        for i, name in enumerate(joint_names):
            jid = self._mj.mj_name2id(
                self.model, self._mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint not found: {name!r}")
            out[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
        return out

    def set_body_pose(self, body_name, position, quaternion=None):
        bid = self._mj.mj_name2id(
            self.model, self._mj.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body not found: {body_name!r}")
        jid  = self.model.body_jntadr[bid]
        qadr = self.model.jnt_qposadr[jid]
        self.data.qpos[qadr:qadr+3] = position
        if quaternion is not None:
            self.data.qpos[qadr+3:qadr+7] = quaternion
        else:
            self.data.qpos[qadr+3] = 1.0
            self.data.qpos[qadr+4:qadr+7] = 0.0
        self.forward()

    def get_body_position(self, body_name):
        bid = self._mj.mj_name2id(
            self.model, self._mj.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body not found: {body_name!r}")
        return self.data.xpos[bid].copy()

    def render_camera(self, camera_name="overhead_cam"):
        if self._renderer is None:
            raise RuntimeError("Use MuJoCoEnv as context manager")
        self._renderer.update_scene(self.data, camera=camera_name)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=camera_name)
        depth_m = self._renderer.render().astype(np.float32)
        self._renderer.disable_depth_rendering()
        # MuJoCo 3.x Renderer.render() in depth mode returns linear metric
        # depths in metres directly (not a z-buffer in [0,1]).
        # Mark pixels at or beyond the far plane as NaN.
        extent = float(self.model.stat.extent)
        zfar_m = float(self.model.vis.map.zfar) * extent
        depth_m[depth_m >= zfar_m * 0.999] = np.nan
        return rgb, depth_m

    def viewer_is_running(self):
        return self._viewer is not None and self._viewer.is_running()

    def sync_viewer(self):
        if self._viewer:
            self._viewer.sync()

    @property
    def dt(self):
        return float(self.model.opt.timestep)

    @property
    def sim_time(self):
        return float(self.data.time)
