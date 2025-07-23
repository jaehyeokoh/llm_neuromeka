import os
import argparse
import torch
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=False, default="checkpoint/checkpoint_detection.tar", help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.08, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', default=False, help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode')
parser.add_argument('--collision_detection', default=True, help='Enable collision_detection mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))


def get_aligned_point_cloud_from_realsense():
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
        colors = np.asanyarray(color_frame.get_data()).astype(np.float32) / 255.0  # HWC

        xmap, ymap = np.meshgrid(np.arange(depths.shape[1]), np.arange(depths.shape[0]))
        points_z = depths * scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        mask = (points_z > 0.01) & (points_z < 1.0)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)



        return points, colors

    finally:
        pipeline.stop()

def angle_between(v1, v2):
    """ 두 벡터 사이의 각도 (라디안) """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot)

def graspnet2robot(R_graspnet, robot_axes=('z', 'x', 'y'), flip_axes=(False, False, False)):
    """
    GraspNet 회전행렬을 로봇 좌표계 기준 회전행렬로 변환한다.

    Args:
        R_graspnet: (3, 3) numpy.ndarray – GraspNet에서 얻은 회전 행렬
        robot_axes: Tuple[str, str, str] – 로봇의 그리퍼 좌표계 (approach, open, vertical) 순서
                    ex: ('z', 'x', 'y') → 로봇은 Z: 접근, X: 벌림, Y: 수직
        flip_axes: Tuple[bool, bool, bool] – 각 축을 반전해야 할 경우 지정 (기본 False)

    Returns:
        R_robot: (3, 3) numpy.ndarray – 로봇 좌표계에 맞춘 회전 행렬
    """
    axes_map = {'x': 0, 'y': 1, 'z': 2}
    R_robot = np.zeros((3, 3))
    graspnet_axes = [R_graspnet[:, i] for i in range(3)]  # [X, Y, Z] 축 벡터

    for i, axis in enumerate(robot_axes):
        sign = -1 if flip_axes[i] else 1
        R_robot[:, i] = sign * graspnet_axes[axes_map[axis]]

    return R_robot


# def graspnet_to_ur_rotation(R_graspnet, correction_axis=None, correction_deg=0):
#     """
#     GraspNet 회전 행렬을 UR 좌표계 기준 (Z, Y, X 순서)로 변환한다.
    
#     GraspNet 기준:
#         - X축: 접근 방향 (approach)
#         - Y축: 벌림 방향 (open)
#         - Z축: palm 방향

#     UR 기준 (Z, Y, X 순서로 사용하는 경우):
#         - Z축: 접근 방향
#         - Y축: palm 방향
#         - X축: open 방향

#     Returns:
#         R_ur (np.ndarray): (3, 3) 회전 행렬 (UR 기준, 열 순서: [Z, Y, X])
#         rpy_deg (tuple): (roll, pitch, yaw) in degrees, UR 기준
#     """
#     # GraspNet → UR 재배치
#     approach = R_graspnet[:, 0]  # GraspNet X
#     open_dir = R_graspnet[:, 1]  # GraspNet Y
#     palm     = R_graspnet[:, 2]  # GraspNet Z

#     # 열 순서: [X, Y, Z] = [open, palm, approach] ⇒ 우리는 ZYX 쓰고 싶으므로:
#     R_ur = np.column_stack((approach, palm, open_dir))  # 열: Z, Y, X

#     # 보정 회전 (optional)
#     if correction_axis is not None and correction_deg != 0:
#         correction_rot = R.from_euler(correction_axis.lower(), correction_deg, degrees=True)
#         R_ur = R_ur @ correction_rot.as_matrix()

#     # Euler 각도 추출 (UR 기준 XYZ → roll, pitch, yaw)
#     rpy_deg = R.from_matrix(R_ur).as_euler('xyz', degrees=True)

#     return R_ur, rpy_deg

def graspnet_to_ur_rotation_matrix(R_graspnet):
    """
    GraspNet 좌표계를 UR 좌표계로 변환하면서 회전 정보(기울기 포함)를 유지하는 함수.
    
    GraspNet: X=approach, Y=open_dir, Z=palm  
    UR 기준:  Z=approach, Y=palm, X=open_dir

    Returns:
        R_ur: UR 기준 회전 행렬
    """

    T = np.array([
        [0, 0, 1],  # UR X = GraspNet y (open_dir)
        [0, 1, 0],  # UR y = GraspNet z (palm)
        [1, 0, 0],  # UR Z = GraspNet X (approach)
    ])

    R_ur = T @ R_graspnet @ T.T
    rpy= R.from_matrix(R_ur).as_euler('zyx', degrees=True)

    # 오프셋 보정 (각도 시작점 다 달라서 이렇게 일단 만듬)
    rpy[0]  = 180 - rpy[0]
    rpy[1] = 90  - rpy[1]
    rpy[2]   = 90  + rpy[2]

    rpy

    return R_ur, rpy



def graspnet_to_ur_rotation(R_graspnet):
    """
    GraspNet 회전 행렬을 UR 좌표계 기준으로 변환 (Z, Y, X 순서) + 오른손 좌표계 보장

    Returns:
        R_ur: (3, 3) 회전 행렬 (UR 기준)
        rpy_deg: (roll, pitch, yaw) in degrees
    """

        # GraspNet 원본 접근 벡터
    approach_graspnet = R_graspnet[:, 0]
    angle_graspnet = np.degrees(np.arcsin(abs(approach_graspnet[2])))  # 기대값: 약 9.4도

    R_ur_rotation_matrix, angle_rotat = graspnet_to_ur_rotation_matrix(R_graspnet)


    approach = R_graspnet[:, 0]  # GraspNet X
    open_dir = R_graspnet[:, 1]  # GraspNet Y
    palm     = R_graspnet[:, 2]  # GraspNet Z

    # 열 순서: [Z, Y, X] = [approach, palm, open]
    R_ur = np.column_stack((approach, palm, open_dir))

    # 오르손 좌표계 보장: det(R) == +1
    if np.linalg.det(R_ur) < 0:
        # X축 반전 (가장 일반적)
        R_ur[:, 0] *= -1

    # UR 기준 접근 벡터
    approach_ur = R_ur[:, 2]
    angle_ur = np.degrees(np.arcsin(abs(approach_ur[2])))  # 문제가 있는 경우: 0도

    rpy_deg_rot = R.from_matrix(R_ur_rotation_matrix).as_euler('zyx', degrees=True)
    rpy_deg_rot_xyz = R.from_matrix(R_ur_rotation_matrix).as_euler('xyz', degrees=True)
    rpy_deg_rot_zxy = R.from_matrix(R_ur_rotation_matrix).as_euler('zxy', degrees=True)
    rpy_deg_rot_yzx = R.from_matrix(R_ur_rotation_matrix).as_euler('yzx', degrees=True)
    print("rpy matrix issisisiis", rpy_deg_rot)
    print("rpy 여러개","xyz",rpy_deg_rot_xyz, "zxy",rpy_deg_rot_zxy, "yzx",  rpy_deg_rot_yzx )

    # 오프셋 보정
    rpy_deg_rot[0]  = 180 - rpy_deg_rot[0]
    rpy_deg_rot[1] = 90  - rpy_deg_rot[1]
    rpy_deg_rot[2]   = 90  + rpy_deg_rot[2]


    print("angle angle_graspnet is ",angle_graspnet,"angle_ur is ",angle_ur, "angle rotation matrix 적용", angle_rotat, "rpy rotation matrix", rpy_deg_rot)
    # Euler 추출
    rpy_deg = R.from_matrix(R_ur).as_euler('xyz', degrees=True)

    return R_ur, rpy_deg

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

    # 🔹 중심 기준 필터링
    center = np.array([0.0, 0.0, 0.0])
    radius = 0.15  # 10cm
    max_angle_deg = 0
    max_angle_rad = np.deg2rad(max_angle_deg)
    z_axis = np.array([0, 0, 1])
    filtered = [
        g for g in gg
        if np.linalg.norm(g.translation[:2] - center[:2]) < radius
        # and abs(angle_between(g.rotation_matrix[:, 0], z_axis)) >= max_angle_rad0
    ]
    # filtered = [g for g in gg if np.linalg.norm(g.translation[:2] - center[:2]) < radius]

    if len(filtered) == 0:
        print("No grasps found near center region.")
        return

    # best_grasp = max(filtered, key=lambda g: g.score)
    # theta_deg = np.rad2deg(angle_between(best_grasp.rotation_matrix[:, 0], z_axis))
    # # print(f"  Approach Angle w.r.t Z (deg) : {theta_deg:.2f}")

    # print(f"Selected Grasp:")
    # print(f"  Score          : {best_grasp.score:.4f}")
    # print(f"  Position (xyz) : {best_grasp.translation}")
    # print(f"  Approach       : {best_grasp.rotation_matrix[:, 0]}")
    # print(f"  Binormal       : {best_grasp.rotation_matrix[:, 1]}")
    # print(f"  Axis           : {best_grasp.rotation_matrix[:, 2]}")




    best_grasp = max(filtered, key=lambda g: g.score)
    R_grasp_camera = best_grasp.rotation_matrix  # AnyGrasp 출력
    
    R_grasp_global = R_grasp_camera
    print("rotation is ",R_grasp_global)






    R_robot, rpy =graspnet_to_ur_rotation(R_grasp_global)

    print("r robot", R_robot, "rpy:",rpy)
    # 접근 벡터 → UR의 Z축
    approach = R_grasp_global[:, 0]
    open_dir = R_grasp_global[:, 1]
    palm     = R_grasp_global[:, 2]

    # UR rotation matrix = [X, Y, Z] = [open, palm, approach]
    R_ur_correct = np.column_stack((open_dir, palm, approach))
    print("r correct", R_ur_correct)


    # UR 좌표계: [X, Y, Z] = [open, palm, approach]
    R_ur = np.column_stack((open_dir, palm, approach))
    from scipy.spatial.transform import Rotation as R
    r = R.from_matrix(R_ur)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)
    print("UR용 RPY:", roll, pitch, yaw)










    R_grasp_global = R_ur_correct

    rot_mat = np.array(R_grasp_global)
    from scipy.spatial.transform import Rotation as R
    # 회전 객체 생성 및 RPY 추출 (ZYX 순서: yaw, pitch, roll)
    r = R.from_matrix(rot_mat)

    roll_s, pitch_s, yaw_s = r.as_euler('xyz', degrees=True) 

    # 변환
    yaw_r   = yaw_s
    pitch_r = pitch_s
    roll_r  = roll_s

    print("Converted to robot:", roll_r, pitch_r, yaw_r)













    theta_deg = np.rad2deg(angle_between(R_grasp_global[:, 0], z_axis))
    print(f"  Approach Angle w.r.t Z (deg) : {theta_deg:.2f}")

    print(f"Selected Grasp:")
    print(f"  Score          : {best_grasp.score:.4f}")
    print(f"  Position (xyz) : {best_grasp.translation}")
    # print(f"  Approach       : {best_grasp.rotation_matrix[:, 0]}")
    # print(f"  Binormal       : {best_grasp.rotation_matrix[:, 1]}")
    # print(f"  Axis           : {best_grasp.rotation_matrix[:, 2]}")
    print(f"  Approach       : {R_grasp_global[:, 0]}")
    print(f"  Binormal       : {R_grasp_global[:, 1]}")
    print(f"  Axis           : {R_grasp_global[:, 2]}")

    print(f"  Width (m)      : {best_grasp.width:.4f}")
    # 회전 행렬 정의
    rot_mat = np.array(R_grasp_global)

    # 회전 객체 생성 및 RPY 추출 (ZYX 순서: yaw, pitch, roll)
    r = R.from_matrix(rot_mat)

    roll_s, pitch_s, yaw_s = r.as_euler('xyz', degrees=True) 

    # 변환
    yaw_r   = yaw_s
    pitch_r = pitch_s
    roll_r  = roll_s

    print("Converted to robot:", roll_r, pitch_r, yaw_r)
    if cfgs.debug:
        trans_mat = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]])
        cloud.transform(trans_mat)

        # 전체 후보군 시각화
        grippers_all = [g.to_open3d_geometry() for g in filtered]
        for g in grippers_all:
            g.transform(trans_mat)

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=0.1)
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color([0.2, 0.7, 0.9])
        cylinder.translate(center - np.array([0, 0, 0.1 / 2]))
        cylinder.transform(trans_mat)
        # 좌표축 시각화용 객체 추가
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        print("[🔎] Showing all grasp candidates...")
        o3d.visualization.draw_geometries([*grippers_all, cloud,axis])




        # === grasp array 분해 ===
        arr = best_grasp.grasp_array
        approach = arr[4:7]
        binormal = arr[7:10]
        axis     = arr[10:13]
        position = arr[13:16]

        # === 시각화 기준 변환행렬 ===
        T = np.diag([1, 1, -1])
        T_h = np.eye(4)
        T_h[:3, :3] = T
        t_viz = T @ position

        # === 시각화용 기준좌표축 ===
        axis_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # === 축 이름별 색상 ===
        colors = {
            'X': [1, 0, 0],   # 빨강
            'Y': [0, 1, 0],   # 초록
            'Z': [0, 0, 1],   # 파랑
            'F': [1, 0, 1],   # 보라 (forward 기준)
        }

        # === 화살표 생성 함수 ===
        def create_arrow(vec, color, origin):
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.002, cone_radius=0.004,
                cylinder_height=0.05, cone_height=0.01
            )
            arrow.paint_uniform_color(color)
            rot, _ = R.align_vectors([[0, 0, 1]], [vec])
            arrow.rotate(rot.as_matrix(), center=(0, 0, 0))
            arrow.translate(origin)
            return arrow

        # === 기준 forward vector 화살표 생성 ===
        R_grasp = best_grasp.rotation_matrix
        forward_vec = T @ (R_grasp @ np.array([1, 0, 0]))
        arrow_forward = create_arrow(forward_vec, colors['F'], t_viz)

        # === 회전행렬 조합들 ===
        rotation_combinations = {
            'A: [approach, binormal, axis]':    [approach, binormal, axis],
            'B: [axis, binormal, approach]':    [axis, binormal, approach],
            'C: [binormal, approach, axis]':    [binormal, approach, axis],
            'D: [approach, axis, binormal]':    [approach, axis, binormal],
            'E: [axis, approach, binormal]':    [axis, approach, binormal],
            'F: [binormal, axis, approach]':    [binormal, axis, approach],
        }

        # === gripper geometry 변환 미리 수행 ===
        gripper_best = best_grasp.to_open3d_geometry()
        gripper_best.transform(T_h)

        # === 실험 루프 ===
        for label, (X_vec, Y_vec, Z_vec) in rotation_combinations.items():
            R_robot = np.column_stack([X_vec, Y_vec, Z_vec])
            
            # 오른손 좌표계 확인
            if np.linalg.det(R_robot) < 0:
                print(f"[⚠️] 조합 '{label}' 은 왼손 좌표계입니다. 스킵합니다.")
                continue

            R_viz = T @ R_robot @ T

            print("\n" + "="*60)
            print(f"[🔍] 실험 조합: {label}")
            print("RPY (deg):", np.round(R.from_matrix(R_viz).as_euler('xyz', degrees=True), 2))
            print("X축:", np.round(R_viz[:, 0], 4))
            print("Y축:", np.round(R_viz[:, 1], 4))
            print("Z축:", np.round(R_viz[:, 2], 4))
            print("기준 Forward:", np.round(forward_vec, 4))
            print("="*60)

            # 각 축 화살표 생성
            arrow_x = create_arrow(R_viz[:, 0], colors['X'], t_viz)
            arrow_y = create_arrow(R_viz[:, 1], colors['Y'], t_viz)
            arrow_z = create_arrow(R_viz[:, 2], colors['Z'], t_viz)

            # 시각화
            print(f"[🎯] 그리퍼 + 조합 '{label}' 화살표 시각화")
            o3d.visualization.draw_geometries([
                cloud, gripper_best, axis_geom,
                arrow_x, arrow_y, arrow_z,
                arrow_forward
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

    # 2) 높이 범위: 물체 최상단(Z_max) ~ Z_max - height_range
    # z_max = cluster_pts[:, 2].max()
    # z_min = z_max - height_range

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


        # 최종 합본 시각화
        # draw_geometries는 리스트 형태의 geometry를 인자로 받습니다.
        # final_pcd가 PointCloud 객체인지 다시 한번 확인해 보세요.
        # o3d.visualization.draw_geometries(
        #     [final_pcd], # 합쳐진 하나의 포인트 클라우드만 전달. 리스트로 감싸야 합니다!
        #     window_name="Merged Clusters and Side Surfaces (Combined)"
        # )
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
    points, colors = get_aligned_point_cloud_from_realsense()
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