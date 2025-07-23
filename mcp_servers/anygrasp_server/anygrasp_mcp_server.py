

import argparse
import torch
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import time


##################################################################3
"""
anygrasp import시 에러 메세지가 나오는데 이게 utf-8형식이 아니라서 mcp에서 오류가 남 그걸 방지하는 코드
"""
import os
import sys

# stdout/stderr 완전 차단 후 gsnet import
original_stdout = sys.stdout
original_stderr = sys.stderr

class NullOutput:
    def write(self, txt): pass
    def flush(self): pass

# gsnet import 전에 출력 차단
sys.stdout = NullOutput()
sys.stderr = NullOutput()

# 환경변수도 추가 설정
os.environ['TERM'] = 'xterm'
os.environ['TERMCAP'] = ''
os.environ['LC_TIME'] = 'C'
os.environ['LC_ALL'] = 'C'

try:
    from gsnet import AnyGrasp
finally:
    # import 완료 후 출력 복원
    sys.stdout = original_stdout
    sys.stderr = original_stderr
###################################################################




from graspnetAPI import GraspGroup
from mcp.server.fastmcp import FastMCP
import base64
import json
import logging
import math
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'

# 모든 print 비활성화
import builtins
original_print = builtins.print
def safe_print(*args, **kwargs):
    try:
        original_print(*[str(arg) for arg in args], **kwargs)
    except:
        pass
builtins.print = safe_print

# ✅ 루트 로거에 달려 있는 기존 핸들러 모두 제거
logging.getLogger().handlers.clear()

# ✅ 전파 방지 (중복 로깅 방지)
logging.getLogger().propagate = False

# ✅ 기본 로그 설정 복구 (필요에 따라 파일 또는 콘솔 설정)
# logging.basicConfig(level=logging.WARNING)  # WARNING 이상 로그만 출력
logging.disable(logging.CRITICAL)

class MathFunc:
    @staticmethod
    def m_to_mm(length):
        """
        length: [m]
        """
        return length * 1000.
    
    @staticmethod
    def mm_to_m(length):
        """
        length: [mm]
        """
        return length / 1000.
 
    @staticmethod
    def degree_to_rad(angle):
        """
        angle: [degree]
        """
        return angle * np.pi / 180.
    
    @staticmethod
    def rad_to_degree(angle):
        """
        angle: [rad]
        """
        return angle * 180. / np.pi
    
    @staticmethod
    def single_axis_rotMat(axis, angle):
        """
        axis: x / y / z
        angle: [rad]
        """
        assert axis in ['x', 'y', 'z'], "Unavailable axis"
        
        c = np.cos(angle)
        s = np.sin(angle)
 
        if axis == 'x':
            return np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]], dtype=np.float32)
        elif axis == 'y':
            return np.array([[c, 0, s],
                            [0, 1, 0],
                            [-s, 0, c]], dtype=np.float32)
        else:
            return np.array([[c, -s, 0],
                            [s, c, 0],
                            [0, 0, 1]], dtype=np.float32)
 
    @staticmethod
    def euler_to_rotMat(euler_x, euler_y, euler_z):
        """
        euler_x, euler_y, euler_z: [rad]
        """
        R_x = MathFunc.single_axis_rotMat('x', euler_x)
        R_y = MathFunc.single_axis_rotMat('y', euler_y)
        R_z = MathFunc.single_axis_rotMat('z', euler_z)
        return R_z @ R_y @ R_x
    
    @staticmethod
    def rotMat_to_euler(rotMat):
        assert(MathFunc.is_rotMat(rotMat)), "Given matrix is not rotation matrix."
        sy = math.sqrt(rotMat[0,0] * rotMat[0,0] + rotMat[1,0] * rotMat[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = math.atan2(rotMat[2,1] , rotMat[2,2])
            y = math.atan2(- rotMat[2,0], sy)
            z = math.atan2(rotMat[1,0], rotMat[0,0])
        else :
            x = math.atan2(- rotMat[1,2], rotMat[1,1])
            y = math.atan2(- rotMat[2,0], sy)
            z = 0
        return np.array([x, y, z])
    
    @staticmethod
    def rotMat_to_quat(R):
        """
        Convert a rotation matrix to a quaternion.
 
        Parameters:
        R (numpy.ndarray): 3x3 rotation matrix.
 
        Returns:
        numpy.ndarray: Quaternion as [w, x, y, z].
        """
        # Ensure the matrix is of the correct shape
        assert R.shape == (3, 3), "Rotation matrix must be 3x3"
 
        # Calculate the trace of the matrix
        trace = np.trace(R)
 
        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
 
        quaternion = np.array([w, x, y, z])
        return quaternion
 
    
    @staticmethod
    def is_rotMat(rotMat):
        Rt = np.transpose(rotMat)
        shouldBeIdentity = np.dot(Rt, rotMat)
        I = np.identity(3, dtype = rotMat.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-4
    
    @staticmethod
    def euler_xyz_to_quat(roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        # compute quaternion
        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp
        return np.array([qw, qx, qy, qz])
    
    @staticmethod
    def quat_to_euler_xyz(quat):
        q_w, q_x, q_y, q_z = quat[0], quat[1], quat[2], quat[3]
        # roll (x-axis rotation)
        sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
        cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
        roll = np.arctan2(sin_roll, cos_roll)
 
        # pitch (y-axis rotation)
        sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
        pitch = np.where(np.abs(sin_pitch) >= 1, np.pi / 2. * np.sign(sin_pitch), np.arcsin(sin_pitch))
 
        # yaw (z-axis rotation)
        sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
        cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
        yaw = np.arctan2(sin_yaw, cos_yaw)
 
        return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)
    
    @staticmethod
    def quat_unique(q):
        return np.where(q[0] < 0, -q, q)
    
    @staticmethod
    def quat_conjugate(q):
        return np.concatenate((q[0:1], -q[1:]))
    
    @staticmethod
    def quat_mul(q1, q2):
        # reshape to (N, 4) for multiplication
        shape = q1.shape
        # extract components from quaternions
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        # perform multiplication
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        return np.array([w, x, y, z])
    
    @staticmethod
    def quat_to_axis_angle(quat, eps: float = 1.0e-6):
        quat = quat * (1.0 - 2.0 * (quat[0:1] < 0.0))
        mag = np.linalg.norm(quat[1:])
        half_angle = np.arctan2(mag, quat[0])
        angle = 2.0 * half_angle
        # check whether to apply Taylor approximation
        sin_half_angles_over_angles = np.where(
            np.abs(angle) > eps, np.sin(half_angle) / angle, 0.5 - angle * angle / 48
        )
        return quat[1:4] / sin_half_angles_over_angles
    
    @staticmethod
    def quat_error_magnitude(q1, q2):
        quat_diff = MathFunc.quat_mul(q1, MathFunc.quat_conjugate(q2))
        return np.linalg.norm(MathFunc.quat_to_axis_angle(quat_diff))
    

# 서버에 고유한 이름 부여
mcp = FastMCP("anygrasp_mcp_server")


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=False, default="checkpoint/checkpoint_detection.tar", help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.09, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.025, help='Gripper height')  # 이거랑 width가 collision detection에 영향 주는듯
parser.add_argument('--top_down_grasp', action='store_true', default=False, help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode')
parser.add_argument('--collision_detection', default=True, help='Enable collision_detection mode')
cfgs = parser.parse_args()
# cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))


def get_point_cloud_from_realsense():
    """ RealSense에서 RGB-D 정렬된 포인트 클라우드 추출 (points, colors 반환) """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    profile = pipeline.start(config)


    align = rs.align(rs.stream.color)

    try:

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        scale = profile.get_device().first_depth_sensor().get_depth_scale()

        for _ in range(30):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Aligned RealSense frame capture failed")

        depths = np.asanyarray(depth_frame.get_data())
        colors = np.asanyarray(color_frame.get_data()).astype(np.float32) / 255.0  # 이미 정규화

        xmap, ymap = np.meshgrid(np.arange(depths.shape[1]), np.arange(depths.shape[0]))
        points_z = depths * scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        mask = (points_z > 0.01) & (points_z < 1.0)
        points = np.stack([points_x, points_y, points_z], axis=-1)

        points_reshaped = points.reshape(-1, 3).astype(np.float32)
        colors_reshaped = colors.reshape(-1, 3).astype(np.float32)

        mask_flat = mask.flatten()
        points = points_reshaped[mask_flat]
        colors = colors_reshaped[mask_flat]



        return points, colors

    finally:
        pipeline.stop()

def angle_between(v1, v2):
    """ 두 벡터 사이의 각도 (라디안) """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot)

def decode_point_cloud(data: dict):
    points = np.frombuffer(base64.b64decode(data["points"]), dtype=np.float32).reshape(-1, 3)
    colors = np.frombuffer(base64.b64decode(data["colors"]), dtype=np.float32).reshape(-1, 3)
    return points, colors


# 전역 AnyGrasp 모델을 로드해서 재사용 (최초 한 번만)
ANYGRASP_MODEL = None

# def move_to_grasp(self, pos_cam_to_gripper_in_cam, rot_cam_to_gripper):
def grasp_rpy( rot_cam_to_gripper):
        
        # Frame: B -> E
        # robot_end_pose = np.array(self.robot.get_state()["p"])
        # pos_BE_in_B = MathFunc.mm_to_m(robot_end_pose[:3])
        robot_end_ori = MathFunc.degree_to_rad(np.array([180.0,0.0,180.0]))
        rot_BE = MathFunc.euler_to_rotMat(
            euler_x=robot_end_ori[0], euler_y=robot_end_ori[1], euler_z=robot_end_ori[2])
        
        # Frame: E -> C
        # Realsense camera frame is attached on the RGB camera center
        # x value is tuned, y and z values are measured
        # pos_EC_in_E = np.array([-0.09, 0.025, 0.035], dtype=np.float32)  # x: -0.095, y: 0.03
        rot_EC = np.array( # End-effector → Camera의 회전행렬
            [[0., 1., 0.],
            [-1., 0., 0.],
            [ 0., 0., 1.]], dtype=np.float32)
 
        # Use value from indyEye. However, y and z are handtuned...
        # pos_EC_in_E = np.array([-0.09457589, 0.025, 0.035], dtype=np.float32)  # x: -0.095, y: 0.03
        # rot_EC = np.array(
        #     [[-0.00736414, 0.99962831, -0.02624865],
        #     [-0.99963224, -0.00804425, -0.02589949],
        #     [ -0.02610098, 0.02604828, 0.99931997]], dtype=np.float32)
        
        # Frame: C -> G
        # pos_CG_in_C = pos_cam_to_gripper_in_cam.astype(np.float32)
        rot_CG = rot_cam_to_gripper.astype(np.float32)
        
        # Frame: G -> E (fixed)
        rot_GE_fixed = np.array(
            [[0., 0., 1.],
            [0., -1., 0.],
            [1., 0., 0.]], dtype=np.float32)
        
        # Solve for Frame: E -> G
        # pos_EG_in_E = pos_EC_in_E + rot_EC @ pos_CG_in_C
        # pos_EG_in_E = rot_EC @ pos_CG_in_C
        rot_EG = rot_EC @ rot_CG
 
        # Solve for Frame: B -> G (***)
        # pos_BG_in_B = pos_BE_in_B + rot_BE @ pos_EG_in_E  # Second target
        rot_BG = rot_BE @ rot_EG  # First & Second target
        z_threshold = -0.03
        # if pos_BG_in_B[2] < z_threshold:  # Ground detected as graspable object
        #     print(f"Grasp height: {pos_BG_in_B[2]} (< {z_threshold})")
        #     return EndType.NO_OBJECT
 
        # Solve for Frame: fixed E (***)
        rot_BGfixedE = rot_BG @ rot_GE_fixed
        #####
        # Project 6D grasp to 4D grasp (translation + yaw)
        euler_xyz = MathFunc.rotMat_to_euler(rot_BGfixedE)
        rpy = np.rad2deg(euler_xyz)
        #####

        print("rpy is 뉴로메카", rpy)
        return [rpy[0],rpy[1],rpy[2]]

# 누로멬 코드
# def move_to_grasp(self, pos_cam_to_gripper_in_cam, rot_cam_to_gripper):
        
#         # Frame: B -> E
#         robot_end_pose = np.array(self.robot.get_state()["p"])
#         pos_BE_in_B = MathFunc.mm_to_m(robot_end_pose[:3])
#         robot_end_ori = MathFunc.degree_to_rad(robot_end_pose[3:])
#         rot_BE = MathFunc.euler_to_rotMat(
#             euler_x=robot_end_ori[0], euler_y=robot_end_ori[1], euler_z=robot_end_ori[2])
        
#         # Frame: E -> C
#         # Realsense camera frame is attached on the RGB camera center
#         # x value is tuned, y and z values are measured
#         # pos_EC_in_E = np.array([-0.09, 0.025, 0.035], dtype=np.float32)  # x: -0.095, y: 0.03
#         # rot_EC = np.array(
#         #     [[0., 1., 0.],
#         #     [-1., 0., 0.],
#         #     [ 0., 0., 1.]], dtype=np.float32)
 
#         # Use value from indyEye. However, y and z are handtuned...
#         pos_EC_in_E = np.array([-0.09457589, 0.025, 0.035], dtype=np.float32)  # x: -0.095, y: 0.03
#         rot_EC = np.array(
#             [[-0.00736414, 0.99962831, -0.02624865],
#             [-0.99963224, -0.00804425, -0.02589949],
#             [ -0.02610098, 0.02604828, 0.99931997]], dtype=np.float32)
        
#         # Frame: C -> G
#         pos_CG_in_C = pos_cam_to_gripper_in_cam.astype(np.float32)
#         rot_CG = rot_cam_to_gripper.astype(np.float32)
        
#         # Frame: G -> E (fixed)
#         rot_GE_fixed = np.array(
#             [[0., 0., 1.],
#             [0., -1., 0.],
#             [1., 0., 0.]], dtype=np.float32)
        
#         # Solve for Frame: E -> G
#         pos_EG_in_E = pos_EC_in_E + rot_EC @ pos_CG_in_C
#         rot_EG = rot_EC @ rot_CG
 
#         # Solve for Frame: B -> G (***)
#         pos_BG_in_B = pos_BE_in_B + rot_BE @ pos_EG_in_E  # Second target
#         rot_BG = rot_BE @ rot_EG  # First & Second target
#         z_threshold = -0.03
#         if pos_BG_in_B[2] < z_threshold:  # Ground detected as graspable object
#             print(f"Grasp height: {pos_BG_in_B[2]} (< {z_threshold})")
#             return EndType.NO_OBJECT
 
#         # Solve for Frame: fixed E (***)
#         rot_BGfixedE = rot_BG @ rot_GE_fixed
#         #####
#         # Project 6D grasp to 4D grasp (translation + yaw)
#         euler_xyz = MathFunc.rotMat_to_euler(rot_BGfixedE)
#         rot_BGfixedE = MathFunc.euler_to_rotMat(euler_x=np.pi, euler_y=0, euler_z=euler_xyz[2]) # gravity aligned (assume camera facing down)
#         #####


# @mcp.tool()
# async def anygrasp_analyze(data, mask_base64=None, mask_shape=None):
#     global ANYGRASP_MODEL
#     points, colors = decode_point_cloud(data)
#     print("Point Cloud Bounds:", points.min(axis=0), points.max(axis=0))

#     # AnyGrasp 모델 최초 1회만 초기화
#     if ANYGRASP_MODEL is None:
#         ANYGRASP_MODEL = AnyGrasp(cfgs)
#         ANYGRASP_MODEL.load_net()
#         ANYGRASP_MODEL.net.eval()  # 모델을 eval 모드로 명시
#         print("AnyGrasp 모델 로드 완료")

#     lims = [-0.5, 0.25, -0.25, 0.3, 0.3, 1.0]

#     # torch.no_grad()로 메모리 사용 최소화
#     with torch.no_grad():
#         gg, cloud = ANYGRASP_MODEL.get_grasp(
#             points,
#             colors,
#             lims=lims,
#             apply_object_mask=True,
#             dense_grasp=False,
#             collision_detection=True
#         )

#     if gg is None or len(gg) == 0:
#         print('No Grasp detected after collision detection!')
#         return "error 1"

#     gg = gg.nms().sort_by_score()
#     gg = gg[0:80]


#     if len(gg) == 0:
#         print("No grasps found near center region.")
#         return "error 2"


#     if cfgs.debug:
#         cloud.transform

#         # 전체 후보군 시각화
#         grippers_all = [g.to_open3d_geometry() for g in gg]
#         for g in grippers_all:
#             g.transform

#         def serialize_mesh(mesh):
#             return {
#                 "vertices": np.asarray(mesh.vertices).tolist(),
#                 "triangles": np.asarray(mesh.triangles).tolist()
#             }

#         def serialize_point_cloud(pcd):
#             if hasattr(pcd, 'to_legacy'):
#                 pcd = pcd.to_legacy()
#             return {
#                 "points": np.asarray(pcd.points).tolist(),
#                 "colors": np.asarray(pcd.colors).tolist() if pcd.has_colors() else None
#             }



#         # 직렬화
#         gripper_json_list = [serialize_mesh(g) for g in grippers_all]
#         cloud_json = serialize_point_cloud(cloud)

#         result_data = {
#             "grippers": gripper_json_list,
#             "cloud": cloud_json
#         }
#         json_str = json.dumps(result_data)


#     round_num = 4  # 원하는 소숫점 자리수로 설정
#     from scipy.spatial.transform import Rotation as R

#     grasp_list = []
#     for g in gg:
#         R = g.rotation_matrix
#         if isinstance(R, torch.Tensor):
#             R = R.detach().cpu().numpy()

#         rpy = grasp_rpy(g.rotation_matrix)

#         grasp = {
#             "score": round(float(g.score), round_num),
#             "translation": [
#                 round(v, round_num) for v in (
#                     g.translation.tolist() if hasattr(g.translation, "tolist") else list(g.translation)
#                 )
#             ],
#             "rpy": 
#                 rpy
            
#         }
#         grasp_list.append(grasp)


#     # 리턴할 딕셔너리를 먼저 만든다
#     result = {
#         "gg": grasp_list,
#         "result_data": json_str
#     }

#     # 그 후 GPU 관련 텐서 제거
#     for g in gg:
#         if hasattr(g, "rotation_matrix") and isinstance(g.rotation_matrix, torch.Tensor):
#             g.rotation_matrix = g.rotation_matrix.detach().cpu().numpy()
#         if hasattr(g, "translation") and isinstance(g.translation, torch.Tensor):
#             g.translation = g.translation.detach().cpu().numpy()
#         if hasattr(g, "score") and isinstance(g.score, torch.Tensor):
#             g.score = float(g.score.detach().cpu())

#     del gg, cloud
#     torch.cuda.empty_cache()

#     return result







@mcp.tool()
async def anygrasp_analyze(data, target_boundary_info=None, debug = True):
    cfgs.debug = debug
    global ANYGRASP_MODEL
    points, colors = decode_point_cloud(data)
    print(f"📊 Full point cloud: {len(points)} points")
    
    # AnyGrasp 모델 초기화
    if ANYGRASP_MODEL is None:
        ANYGRASP_MODEL = AnyGrasp(cfgs)
        ANYGRASP_MODEL.load_net()
        ANYGRASP_MODEL.net.eval()
        print("AnyGrasp 모델 로드 완료")
    
    lims = [-0.5, 0.5, -0.25, 0.3, 0.3, 1.0]
    
    # 전체 포인트 클라우드로 AnyGrasp 실행
    with torch.no_grad():
        gg, cloud = ANYGRASP_MODEL.get_grasp(
            points,
            colors,
            lims=lims,
            apply_object_mask=False,  # 마스크는 나중에 직접 적용
            dense_grasp=False,
            collision_detection=True
        )
    
    if gg is None or len(gg) == 0:
        print('No Grasp detected after collision detection!')
        return "error 1"
    
    gg = gg.nms().sort_by_score()
    print(f"📊 Total grasps after NMS: {len(gg)}")
    

    # 🔥 target_side_pts 기반 필터링
    if target_boundary_info:
        try:
            import pickle
            boundary_info = pickle.loads(base64.b64decode(target_boundary_info))
            
            # 문자열을 float로 변환
            hull_vertices = [[float(x), float(y)] for x, y in boundary_info["hull_vertices"]]
            z_range = boundary_info["z_range"]
            
            # point-in-polygon 함수
            def point_in_polygon(point, polygon):
                x, y = point
                n = len(polygon)
                inside = False
                
                p1x, p1y = polygon[0]
                for i in range(1, n + 1):
                    p2x, p2y = polygon[i % n]
                    if y > min(p1y, p2y):
                        if y <= max(p1y, p2y):
                            if x <= max(p1x, p2x):
                                if p1y != p2y:
                                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                    p1x, p1y = p2x, p2y
                return inside
            
            # 필터링
            filtered_gg = []
            for g in gg:
                pos = g.translation
                if hasattr(pos, 'tolist'):
                    pos = pos.tolist()
                elif hasattr(pos, 'detach'):
                    pos = pos.detach().cpu().numpy()
                
                x, y, z = pos[0], pos[1], pos[2]
                
                if (z_range["min"] <= z <= z_range["max"] and 
                    point_in_polygon([x, y], hull_vertices)): # z범위 지정하려고 했는데 이상하게 나옴
                # if (point_in_polygon([x, y], hull_vertices)):
                    filtered_gg.append(g)
            
            gg = filtered_gg if len(filtered_gg) > 0 else gg
            
        except Exception as e:
            pass  # 실패시 기존 grasp 유지




    print(f"📊 Final: {len(gg)} grasps after mask filtering + top 40")
    
    if len(gg) == 0:
        return "error 2"
    
    json_str = None
    
    # 시각화 및 직렬화 (기존과 동일)
    if cfgs.debug:
        grippers_all = [g.to_open3d_geometry() for g in gg]
    
        def serialize_mesh(mesh):
            return {
                "vertices": np.asarray(mesh.vertices).tolist(),
                "triangles": np.asarray(mesh.triangles).tolist()
            }
        
        def serialize_point_cloud(pcd):
            if hasattr(pcd, 'to_legacy'):
                pcd = pcd.to_legacy()
            return {
                "points": np.asarray(pcd.points).tolist(),
                "colors": np.asarray(pcd.colors).tolist() if pcd.has_colors() else None
            }
        
        gripper_json_list = [serialize_mesh(g) for g in grippers_all] if cfgs.debug else []
        cloud_json = serialize_point_cloud(cloud)
        result_data = {
            "grippers": gripper_json_list,
            "cloud": cloud_json
        }
        json_str = json.dumps(result_data)
    
    # Grasp 변환
    round_num = 4
    grasp_list = []
    for g in gg:
        R_matrix = g.rotation_matrix
        if isinstance(R_matrix, torch.Tensor):
            R_matrix = R_matrix.detach().cpu().numpy()
        rpy = grasp_rpy(g.rotation_matrix)
        grasp = {
            "score": round(float(g.score), round_num),
            "translation": [
                round(v, round_num) for v in (
                    g.translation.tolist() if hasattr(g.translation, "tolist") else list(g.translation)
                )
            ],
            "rpy": rpy
        }
        grasp_list.append(grasp)
    
    result = {
        "gg": grasp_list,
        "result_data": json_str
    }
    
    # GPU 텐서 정리
    for g in gg:
        if hasattr(g, "rotation_matrix") and isinstance(g.rotation_matrix, torch.Tensor):
            g.rotation_matrix = g.rotation_matrix.detach().cpu().numpy()
        if hasattr(g, "translation") and isinstance(g.translation, torch.Tensor):
            g.translation = g.translation.detach().cpu().numpy()
        if hasattr(g, "score") and isinstance(g.score, torch.Tensor):
            g.score = float(g.score.detach().cpu())
    
    del gg, cloud
    torch.cuda.empty_cache()
    
    return result




async def demo():
    points, colors = get_point_cloud_from_realsense()
    await anygrasp_analyze(points, colors)


# if __name__ == '__main__':
#     try:
#         demo()
#     except Exception as e:
#         print(e)

# # --- MCP 서버 시작 ---
if __name__ == "__main__":
    print("MCP 서버 루프 시작...")
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\n사용자에 의해 서버 중지됨 (Ctrl+C).")
    except Exception as e:
        print(f"\nMCP 서버 오류 발생: {e}")

