import os
import argparse
import torch
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

import math
import numpy as np
from scipy.spatial.transform import Rotation as R



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
    



parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=False, default="checkpoint/checkpoint_detection.tar", help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.08, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', default=False, help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode')
parser.add_argument('--collision_detection', default=True, help='Enable collision_detection mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))



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

        # points = points_reshaped[mask].astype(np.float32)
        # colors = colors_reshaped[mask].astype(np.float32)


        return points, colors

    finally:
        pipeline.stop()


# def move_to_grasp(self, pos_cam_to_gripper_in_cam, rot_cam_to_gripper):
def move_to_grasp( rot_cam_to_gripper):
        
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
        rot_EC = np.array(
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
        rot_BGfixedE = MathFunc.euler_to_rotMat(euler_x=np.pi, euler_y=0, euler_z=euler_xyz[2]) # gravity aligned (assume camera facing down)
        #####

        print("rpy is 뉴로메카", rpy)



def angle_between(v1, v2):
    """ 두 벡터 사이의 각도 (라디안) """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot)


def analyze_point_cloud_with_anygrasp(points, colors):
    print("Point Cloud Bounds:", points.min(axis=0), points.max(axis=0))

    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    lims = [-0.5, 0.25, -0.25, 0.3, 0.3, 1.0]

    gg, cloud = anygrasp.get_grasp(
        points,
        colors,
        lims=lims,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True
    )

    if gg is None or len(gg) == 0:
        print('No Grasp detected after collision detection!')
        return

    gg = gg.nms().sort_by_score()


    

    # print("받은거로 구한거 ",rotmat_to_rpy_deg(R_grasp_global))










    # 🔹 중심 기준 필터링
    center = np.array([0.0, 0.0, 0.0])
    radius = 0.15  # 10cm
    z_axis = np.array([0, 0, 1])
    best_grasp = max(gg, key=lambda g: g.score)
    R_grasp_camera = best_grasp.rotation_matrix  # AnyGrasp 출력
    
    R_grasp_global = R_grasp_camera
    print("rotation is ",R_grasp_global)


    rot_mat = np.array(R_grasp_global)


    ############################################################################3

    move_to_grasp(R_grasp_global)


#############################################################################3
    from scipy.spatial.transform import Rotation as R
    # 회전 객체 생성 및 RPY 추출 (ZYX 순서: yaw, pitch, roll)
    r = R.from_matrix(rot_mat)

    roll_s, pitch_s, yaw_s = r.as_euler('zyx', degrees=True) 


    print("Converted to robot_ zyx:", roll_s, pitch_s, yaw_s)

    # theta_deg = np.rad2deg(angle_between(R_grasp_global[:, 0], z_axis))
    # print(f"  Approach Angle w.r.t Z (deg) : {theta_deg:.2f}")

    # print(f"Selected Grasp:")
    # print(f"  Score          : {best_grasp.score:.4f}")
    # print(f"  Position (xyz) : {best_grasp.translation}")
    # # print(f"  Approach       : {best_grasp.rotation_matrix[:, 0]}")
    # # print(f"  Binormal       : {best_grasp.rotation_matrix[:, 1]}")
    # # print(f"  Axis           : {best_grasp.rotation_matrix[:, 2]}")
    # print(f"  Approach       : {R_grasp_global[:, 0]}")
    # print(f"  Binormal       : {R_grasp_global[:, 1]}")
    # print(f"  Axis           : {R_grasp_global[:, 2]}")

    # print(f"  Width (m)      : {best_grasp.width:.4f}")
    # 회전 행렬 정의
    rot_mat = np.array(R_grasp_global)

    # 회전 객체 생성 및 RPY 추출 (ZYX 순서: yaw, pitch, roll)
    r = R.from_matrix(rot_mat)
    if cfgs.debug:
        cloud

        # # 전체 후보군 시각화
        grippers_all = [g.to_open3d_geometry() for g in gg]

        # 좌표축 시각화용 객체 추가
        # 좌표축 객체 만들기
        axis_robot = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        axis_robot.transform


        gripper_best = best_grasp.to_open3d_geometry()
        print("[⭐] Showing best grasp only...")
        o3d.visualization.draw_geometries([gripper_best, cloud,axis_robot])
        # 기존 좌표축: GraspNet 기준
        axis_graspnet = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        # 복사해서 로봇 좌표계용 좌표축 생성
        axis_robot.translate([ 0.05, 0, 0])     # 오른쪽으로 5cm`


        print("[🔎] Showing all grasp candidates with both coordinate systems...")

        o3d.visualization.draw_geometries([
            *grippers_all,
            cloud,
            axis_graspnet,  # 원본 좌표계
            axis_robot       # 변환된 좌표계
        ])


def draw_cluster_contour_on_xy(cluster_pts, avg_z):
    # XY만 뽑아서 2D convex hull
    pts2d = cluster_pts[:, :2]
    if pts2d.shape[0] < 3:
        return None
    hull = ConvexHull(pts2d)
    verts = hull.vertices
    # 다시 3D로 (Z = avg_z)
    hull3d = np.column_stack([pts2d[verts], np.full(len(verts), avg_z)])
    lines = [[i, (i+1) % len(verts)] for i in range(len(verts))]
    colors = [[1, 0, 0] for _ in lines]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(hull3d),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def generate_side_point_cloud(cluster_pts: np.ndarray,
                              num_edge_samples: int = 30,
                              num_height_samples: int = 30,
                              height_range: float = 0.1) -> np.ndarray:
    """
    cluster_pts: (N,3) 해당 클러스터의 원본 포인트
    height_range: 최상단에서 아래로 샘플링할 높이 (m)
    """
    # 1) XY 평면에서 Convex Hull (윗면 윤곽)
    pts2d = cluster_pts[:, :2]
    hull = ConvexHull(pts2d)
    hull_xy = pts2d[hull.vertices]

    # z_min = cluster_pts[:, 2].min()
    zs = cluster_pts[:, 2]
    # 20번째로 작은 Z를 바닥 높이로 사용
    z_min = np.partition(zs, 100)[100]
    z_max =  z_min + height_range

    # 3) 옆면 샘플링
    side_pts = []
    for i in range(len(hull_xy)):
        p1 = hull_xy[i]
        p2 = hull_xy[(i+1)%len(hull_xy)]
        # 변을 따라 샘플링
        for t in np.linspace(0, 1, num_edge_samples):
            x, y = p1*(1-t) + p2*t
            # 높이 방향으로 샘플링
            for z in np.linspace(z_max, z_min, num_height_samples):
                side_pts.append([x, y, z])

    return np.array(side_pts)




import time
from contextlib import contextmanager

import numpy as np
import cupy as cp
import open3d as o3d
from cuml.cluster import DBSCAN as cuDBSCAN
from scipy.spatial import ConvexHull

@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    print(f"[TIMER] {name}: {time.perf_counter() - start:.3f} s")




def cluster_and_visualize_with_volume_filter(pts: np.ndarray,
                                             cols: np.ndarray,
                                             eps: float = 0.005,
                                             min_samples: int = 10,
                                             min_volume: float = 1e-6):
    
    # 0) pts를 GPU로 한 번만 올리기
    with timer("NumPy→CuPy"):
        pts_gpu = cp.asarray(pts)

    # 1) GPU-accelerated DBSCAN (GPU 상에서만)
    with timer("cuML DBSCAN"):
        cu_db = cuDBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels_gpu = cu_db.fit_predict(pts_gpu)

    # 2) 결과를 한 번만 CPU로 복사
    with timer("CuPy→NumPy"):
        labels = cp.asnumpy(labels_gpu)

    max_label = labels.max()
    print(f"[DBSCAN] Detected {max_label + 1} clusters (+ noise)")

    # 이하 기존 로직: 전체/군집별 시각화 & 볼륨 필터링
    def compute_convex_hull_volume(arr):
        if arr.shape[0] < 4: return 0.0
        return ConvexHull(arr).volume

    # 전체 시각화 (생략 가능)
    with timer("전체 시각화"):
        pcd_all = o3d.geometry.PointCloud()
        pcd_all.points = o3d.utility.Vector3dVector(pts)
        pcd_all.colors = o3d.utility.Vector3dVector(cols)
        o3d.visualization.draw_geometries([pcd_all], window_name="All")


        # 최종 병합용 리스트
    merged_cluster_pts = []
    merged_cluster_cols = []
    merged_side_pts = []


    # 군집별 필터링 & 시각화
    with timer("군집 볼륨 계산·시각화"):
        for lbl in range(max_label + 1):
            mask = labels == lbl
            if not np.any(mask): continue

            cluster_pts = pts[mask]
            vol = compute_convex_hull_volume(cluster_pts)
            print(f" Cluster {lbl}: {vol:.3e} m³")
            if vol < min_volume:
                print("  → Skipped")
                continue
            # 1) 병합 리스트에 추가
            merged_cluster_pts.append(cluster_pts)
            merged_cluster_cols.append(cols[mask])


            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cluster_pts)
            pcd.colors = o3d.utility.Vector3dVector(cols[mask])
            # o3d.visualization.draw_geometries([pcd], window_name=f"Cluster {lbl}")
            # 평균 Z값 계산
            avg_z = float(cluster_pts[:, 2].mean())
            # 윤곽선 LineSet 생성
            contour_ls = draw_cluster_contour_on_xy(cluster_pts, avg_z)

            # 시각화 리스트에 contour_ls 추가
            to_draw = [pcd]
            if contour_ls is not None:
                to_draw.append(contour_ls)

            # o3d.visualization.draw_geometries(
            #     to_draw,
            #     window_name=f"Cluster {lbl} with Contour"
            # )
            # (2) 옆면 포인트 생성
            side_pts = generate_side_point_cloud(cluster_pts,
                                                num_edge_samples=30,
                                                num_height_samples=30)
            merged_side_pts.append(side_pts)


            # (3) Open3D로 시각화
            pcd_side = o3d.geometry.PointCloud()
            pcd_side.points = o3d.utility.Vector3dVector(side_pts)
            # (원하시면 색을 지정하거나 원본 cols를 변형해서 씌우셔도 됩니다)
            pcd_side.paint_uniform_color([0.8,0.2,0.2])  # 예: 붉은 옆면

            # # # (4) 기존 점군 + 옆면 점군 같이 보기
            # o3d.visualization.draw_geometries(
            #     [pcd, pcd_side],
            #     window_name=f"Cluster {lbl} Side Surface"
            # )


    # 루프 끝난 뒤 한 번에 합치기
    if merged_cluster_pts:
        all_pts = np.vstack(merged_cluster_pts)
        all_cols = np.vstack(merged_cluster_cols)
        merged_side = np.vstack(merged_side_pts) if merged_side_pts else np.empty((0,3))

        # 병합된 원본 클러스터
        pcd_merged = o3d.geometry.PointCloud()
        pcd_merged.points = o3d.utility.Vector3dVector(all_pts)
        pcd_merged.colors = o3d.utility.Vector3dVector(all_cols)

        # 병합된 옆면
        pcd_side_merged = o3d.geometry.PointCloud()
        pcd_side_merged.points = o3d.utility.Vector3dVector(merged_side)
        pcd_side_merged.paint_uniform_color([0.8,0.2,0.2])



        # 최종 합본 시각화
        # o3d.visualization.draw_geometries(
        #     [pcd_merged, pcd_side_merged],
        #     window_name="Merged Clusters and Side Surfaces"
        # )
        
        # 초기 final_pcd를 빈 PointCloud로 시작
        final_pcd = o3d.geometry.PointCloud()

        # pcd_merged에 포인트가 있다면 final_pcd에 추가
        if len(pcd_merged.points) > 0:
            final_pcd += pcd_merged

        # pcd_side_merged에 포인트가 있다면 final_pcd에 추가
        if len(pcd_side_merged.points) > 0:
            final_pcd += pcd_side_merged

        return final_pcd
    else:
        print("필터 통과한 군집이 없습니다.")



def get_point_cloud_as_numpy(pcd: o3d.geometry.PointCloud):
    """
    Open3D PointCloud 객체를 NumPy 배열 (points, colors)로 변환하여 반환합니다.
    색상은 0.0 ~ 1.0 범위의 float32로 유지됩니다.

    Args:
        pcd (o3d.geometry.PointCloud): 변환할 포인트 클라우드 객체.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - points: (N, 3) 형태의 numpy.ndarray (float32), 점의 3D 좌표.
            - colors: (N, 3) 형태의 numpy.ndarray (float32), 점의 RGB 색상 (0.0~1.0).
    """
    points_np = np.asarray(pcd.points, dtype=np.float32)
    colors_np = np.asarray(pcd.colors, dtype=np.float32)
    return points_np, colors_np


def demo():
    points, colors = get_point_cloud_from_realsense()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(points)
    pcd_all.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_all,axis], window_name="All")


    H = points[:, 2]
    idx_near10 = np.argpartition(H, 10)[:10]
    tenth_idx = idx_near10[np.argmax(H[idx_near10])]
    H10 = H[tenth_idx]
    mask = (H >= H10) & (H <= H10 + 0.1)
    sel_pts = points[mask]
    sel_cols = colors[mask]
    point_generated = cluster_and_visualize_with_volume_filter(sel_pts, sel_cols)


    # 초기 전체 클라우드 (시각화를 위해 Open3D PointCloud 객체로 변환)
    pcd_initial = o3d.geometry.PointCloud()
    pcd_initial.points = o3d.utility.Vector3dVector(points)
    pcd_initial.colors = o3d.utility.Vector3dVector(colors)

    if len(point_generated.points) > 0:
        # pcd_initial의 포인트 중 processed_pcd에 포함되지 않은 부분만 남기고 싶다면 더 복잡한 로직이 필요합니다.
        # 여기서는 단순히 두 개의 PointCloud를 합칩니다.
        final_visualization_pcd = pcd_initial + point_generated
    else:
        final_visualization_pcd = pcd_initial
        print("처리된 클러스터가 없어 초기 클라우드만 표시합니다.")

    # 최종 결과 시각화
    # o3d.visualization.draw_geometries(
    #     [final_visualization_pcd],
    #     window_name="Initial + Processed Clusters"
    # )
    # 좌표축 시각화용 객체 추가
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # 최종 시각화
    o3d.visualization.draw_geometries(
        [final_visualization_pcd, axis],  # 축 추가
        window_name="Initial + Processed Clusters with Global Axis"
    )
    # 시각화 이후, final_visualization_pcd를 NumPy 배열로 변환
    final_points_np, final_colors_np = get_point_cloud_as_numpy(final_visualization_pcd)
    analyze_point_cloud_with_anygrasp(final_points_np, final_colors_np)


if __name__ == '__main__':
    demo()



