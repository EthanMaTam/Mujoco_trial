import mujoco
import mujoco.viewer
from sim.mj_world import MjWorld, debug_print_actuators
from control.pinocchio_ik import ik_from_xyz_rpy, solve_ik_se3
from control.tsid_ik import TSIDIKSolver
import numpy as np

from control.hand_kin import HandSynergyModel, HandIKSolver, HandKinematics, build_default_gesture_space, \
    build_default_hand_synergy


SCENE_PATH = "ur5e_shadowhand_scene.xml"
MJCF_PATH = "ur5e_shadowhand_scene_pin.xml"
HAND_EE = "grasp_site"

def main():

    with MjWorld(xml_path=SCENE_PATH) as world:
        # while world.is_running():
            ik_solver = TSIDIKSolver(use_mjcf=True,
                        mjcf_path=MJCF_PATH,
                        ee_frame_name=HAND_EE,
                        dt=world.dt,
                        q_posture_default=world.default_main_conf,
                        max_iters_default=20000,
                        kp_ee=600,
                        kp_posture=np.array([20, 20, 15, 10, 5, 3, 100, 100], dtype=float),
                        # kp_posture=0
                        )
            world.hold_default_conf()
            world.step()
            gesture = {"grasp": 1.0, "thumb_opposition": 0.8}
            
            
            # b_aabb = world.get_body_aabb_in_body_frame("banana1", include_subtree=True, exact_mesh=False)
            grasp_info = world.propose_grasps_from_bbox("cylinder1")
            
            target = grasp_info.candidates[0].T_w_palm
            target_xyzrpy = grasp_info.candidates[0].xyzrpy_w_palm
            q_sol, ok = ik_solver.solve_ik(pose6=target_xyzrpy, q_init=world.main_qpos)
            
            path = world.main_motion_plan(q_goal_main=q_sol)
            traj_m = world.generate_waypoints_trajectory(path)
            world.follow_trajectory_kinematic(traj_m)
            hand_kin = HandKinematics(MJCF_PATH, HAND_EE)
            hand_synergy = build_default_hand_synergy(hand_kin, base_q_full=world.full_qpos)
            hand_gesture_space = build_default_gesture_space(hand_synergy)
            hand_q_sol = hand_gesture_space.decode_gesture(hand_kin.model_full, world.full_qpos, gesture)
            traj_h = world.generate_segment_trajectory(hand_q_sol)
            world.follow_trajectory_actuator(traj_h)
            # print("----------target----------")
            # print(target)
            # print("----------actually---------")

            # sid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
            # print(world.data.site_xpos[sid])
            # print(world.data.site_xmat[sid].reshape(3, 3))
            # q_meas = world.main_qpos
            # print("q_err(rad):", q_sol - q_meas)
            # world.print_actuator_infos()
        #     hand_q_sol = hand_gesture_space.decode_gesture(hand_kin.model_full, world.full_qpos, gesture)
        #     hand_traj = world.generate_segment_trajectory(hand_q_sol)
        #     world.follow_trajectory_kinematic(hand_traj)
            while world.is_running():
                 world.step()
            # world.save_current_qpos()

            


if __name__ == '__main__':
    main()
    

