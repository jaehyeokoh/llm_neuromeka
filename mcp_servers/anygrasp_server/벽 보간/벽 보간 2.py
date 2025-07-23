import os
import argparse
import torch
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

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
        and abs(angle_between(g.rotation_matrix[:, 0], z_axis)) >= max_angle_rad
    ]
    # filtered = [g for g in gg if np.linalg.norm(g.translation[:2] - center[:2]) < radius]

    if len(filtered) == 0:
        print("No grasps found near center region.")
        return

    best_grasp = max(filtered, key=lambda g: g.score)
    theta_deg = np.rad2deg(angle_between(best_grasp.rotation_matrix[:, 0], z_axis))
    print(f"  Approach Angle w.r.t Z (deg) : {theta_deg:.2f}")

    print(f"Selected Grasp:")
    print(f"  Score          : {best_grasp.score:.4f}")
    print(f"  Position (xyz) : {best_grasp.translation}")
    print(f"  Approach       : {best_grasp.rotation_matrix[:, 0]}")
    print(f"  Binormal       : {best_grasp.rotation_matrix[:, 1]}")
    print(f"  Axis           : {best_grasp.rotation_matrix[:, 2]}")
    print(f"  Width (m)      : {best_grasp.width:.4f}")

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

        print("[🔎] Showing all grasp candidates...")
        o3d.visualization.draw_geometries([*grippers_all, cloud, cylinder])

        # 최고 점수 하나만 다시 강조
        gripper_best = best_grasp.to_open3d_geometry()
        gripper_best.transform(trans_mat)
        print("[⭐] Showing best grasp only...")
        o3d.visualization.draw_geometries([gripper_best, cloud, cylinder])




from sklearn.cluster import DBSCAN
import colorsys
from scipy.spatial import ConvexHull

# def cluster_and_visualize(pts: np.ndarray,
#                           cols: np.ndarray,
#                           eps: float = 0.005,
#                           min_samples: int = 10):
#     """
#     sel_pts, sel_cols에 대해 DBSCAN으로 클러스터링 후
#     colorsys를 이용해 HSV→RGB 컬러맵을 생성해 시각화합니다.
#     """
#     # 1) DBSCAN 수행
#     db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
#     labels = db.fit_predict(pts)  # -1은 noise

#     max_label = labels.max()
#     print(f"[DBSCAN] Detected {max_label + 1} clusters (+ noise)")

#     # 2) HSV 기반 컬러맵 생성
#     n_clusters = max_label + 1
#     # 클러스터마다 고유 Hue (0~1)를 균등 분배
#     hues = np.linspace(0, 1, n_clusters, endpoint=False)
#     rgb_map = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues]
#     # noise 색상: 검정
#     noise_color = (0.0, 0.0, 0.0)

#     # 3) pts 개수만큼 컬러 어레이 채우기
#     colors_mapped = np.zeros((pts.shape[0], 3), dtype=np.float64)
#     for lbl in range(n_clusters):
#         colors_mapped[labels == lbl] = rgb_map[lbl]
#     colors_mapped[labels < 0] = noise_color

#     # 4) Open3D 시각화
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts)
#     pcd.colors = o3d.utility.Vector3dVector(colors_mapped)
#     o3d.visualization.draw_geometries(
#         [pcd],
#         window_name=f"DBSCAN Clustering (HSV colormap): eps={eps}, min_samples={min_samples}"
#     )





# def cluster_and_visualize_with_original_colors(pts: np.ndarray,
#                                                cols: np.ndarray,
#                                                eps: float = 0.005,
#                                                min_samples: int = 10):
#     """
#     pts, cols (RGB [0~1])를 받아 DBSCAN으로 군집화한 뒤,
#     1) 전체를 원본 색으로
#     2) 각 클러스터만 골라 원본 색으로
#     시각화합니다.
#     """
#     # 1) DBSCAN 수행
#     db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
#     labels = db.fit_predict(pts)  # -1은 noise
#     max_label = labels.max()
#     print(f"[DBSCAN] Detected {max_label + 1} clusters (+ noise)")

#     # 2) 전체 원본 색상 시각화
#     pcd_all = o3d.geometry.PointCloud()
#     pcd_all.points = o3d.utility.Vector3dVector(pts)
#     pcd_all.colors = o3d.utility.Vector3dVector(cols)
#     o3d.visualization.draw_geometries(
#         [pcd_all],
#         window_name="All Points (Original Colors)"
#     )

#     # 3) 군집별로 분리해서 원본 색으로 시각화
#     for lbl in range(max_label + 1):
#         mask = labels == lbl
#         if not np.any(mask):
#             continue
#         pcd_cluster = o3d.geometry.PointCloud()
#         pcd_cluster.points = o3d.utility.Vector3dVector(pts[mask])
#         pcd_cluster.colors = o3d.utility.Vector3dVector(cols[mask])
#         o3d.visualization.draw_geometries(
#             [pcd_cluster],
#             window_name=f"Cluster {lbl} (Original Colors)"
#         )


############################################################################## 매우 잘됨
# def cluster_and_visualize_with_volume_filter(pts: np.ndarray,
#                                              cols: np.ndarray,
#                                              eps: float = 0.005,
#                                              min_samples: int = 10,
#                                              min_volume: float = 1e-5):
#     """
#     pts, cols로 DBSCAN 클러스터링 후
#     • 전체 원본 색상 시각화
#     • 각 군집 중 체적 >= min_volume 인 것만 골라 원본 색상으로 시각화
#     """
#     def compute_convex_hull_volume(pts: np.ndarray) -> float:
#         """
#         SciPy ConvexHull을 이용해 pts의 볼록껍질 체적을 계산.
#         점이 4개 미만이면 0 반환.
#         """
#         if pts.shape[0] < 4:
#             return 0.0
#         hull = ConvexHull(pts)
#         return hull.volume  # m^3 단위

#     # 1) DBSCAN 수행
#     db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
#     labels = db.fit_predict(pts)  # –1은 noise
#     max_label = labels.max()
#     print(f"[DBSCAN] Detected {max_label + 1} clusters (+ noise)")

#     # 2) 전체 원본 색상 시각화
#     pcd_all = o3d.geometry.PointCloud()
#     pcd_all.points = o3d.utility.Vector3dVector(pts)
#     pcd_all.colors = o3d.utility.Vector3dVector(cols)
#     o3d.visualization.draw_geometries([pcd_all],
#         window_name="All Points (Original Colors)")

#     # 3) 군집별로 분리, 부피 필터링 후 시각화
#     for lbl in range(max_label + 1):
#         mask = labels == lbl
#         if not np.any(mask):
#             continue

#         # ——— 여기부터 추가된 부분 ———
#         cluster_pts = pts[mask]
#         vol = compute_convex_hull_volume(cluster_pts)
#         print(f" Cluster {lbl}: volume = {vol:.3e} m^3")
#         if vol < min_volume:
#             print(f"  → Skipped (volume below {min_volume:.3e})")
#             continue
#         # ————————————————————————

#         # 필터 통과한 군집만 시각화
#         pcd_cluster = o3d.geometry.PointCloud()
#         pcd_cluster.points = o3d.utility.Vector3dVector(cluster_pts)
#         pcd_cluster.colors = o3d.utility.Vector3dVector(cols[mask])
#         o3d.visualization.draw_geometries(
#             [pcd_cluster],
#             window_name=f"Cluster {lbl} (vol={vol:.3e} m^3)"
#         )




# import time
# from contextlib import contextmanager
# import numpy as np
# import open3d as o3d
# from sklearn.cluster import DBSCAN
# from scipy.spatial import ConvexHull

# @contextmanager
# def timer(name: str):
#     start = time.perf_counter()
#     yield
#     elapsed = time.perf_counter() - start
#     print(f"[TIMER] {name}: {elapsed:.3f} s")

# def cluster_and_visualize_with_volume_filter(pts: np.ndarray,
#                                              cols: np.ndarray,
#                                              eps: float = 0.005,
#                                              min_samples: int = 10,
#                                              min_volume: float = 1e-5):
#     """
#     pts, cols로 DBSCAN 클러스터링 후
#     • 전체 원본 색상 시각화
#     • 각 군집 중 체적 >= min_volume 인 것만 골라 원본 색상으로 시각화
#     (주요 단계별 소요 시간 출력)
#     """
#     def compute_convex_hull_volume(pts: np.ndarray) -> float:
#         if pts.shape[0] < 4:
#             return 0.0
#         hull = ConvexHull(pts)
#         return hull.volume

#     # 1) DBSCAN 수행
#     with timer("DBSCAN"):
#         db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
#         labels = db.fit_predict(pts)  # –1은 noise
#     max_label = labels.max()
#     print(f"[DBSCAN] Detected {max_label + 1} clusters (+ noise)")

#     # 2) 전체 원본 색상 시각화 (이 단계가 너무 오래 걸리면 빼셔도 됩니다)
#     with timer("전체 시각화"):
#         pcd_all = o3d.geometry.PointCloud()
#         pcd_all.points = o3d.utility.Vector3dVector(pts)
#         pcd_all.colors = o3d.utility.Vector3dVector(cols)
#         # o3d.visualization.draw_geometries([pcd_all],
#         #     window_name="All Points (Original Colors)")

#     # 3) 군집별로 분리, 부피 필터링 후 시각화
#     with timer("군집별 볼륨 계산 및 시각화"):
#         for lbl in range(max_label + 1):
#             mask = labels == lbl
#             if not np.any(mask):
#                 continue

#             cluster_pts = pts[mask]
#             vol = compute_convex_hull_volume(cluster_pts)
#             print(f" Cluster {lbl}: volume = {vol:.3e} m^3")
#             if vol < min_volume:
#                 print(f"  → Skipped (volume below {min_volume:.3e})")
#                 continue

#             pcd_cluster = o3d.geometry.PointCloud()
#             pcd_cluster.points = o3d.utility.Vector3dVector(cluster_pts)
#             pcd_cluster.colors = o3d.utility.Vector3dVector(cols[mask])
#             # o3d.visualization.draw_geometries(
#             #     [pcd_cluster],
#             #     window_name=f"Cluster {lbl} (vol={vol:.3e} m^3)"
#             # )


# def demo():
#     # 1) 포인트 클라우드 취득
#     points, colors = get_aligned_point_cloud_from_realsense()

#     # 2) '높이' H는 Z축 값, 카메라에 가까울수록 작다고 간주
#     H = points[:, 2]

#     # 3) Z가 작은 순서로 10번째 값(H10) O(n) 연산으로 찾기
#     # np.argpartition(H, 10)[:10]는 Z가 작은(카메라에 가까운) 10개 인덱스
#     idx_near10 = np.argpartition(H, 10)[:10]
#     # 그 10개 중 가장 큰 Z값이 10번째로 작은 Z
#     tenth_idx = idx_near10[np.argmax(H[idx_near10])]
#     H10 = H[tenth_idx]

#     # 4) H10에서 아래로(카메라에서 멀어지는 방향, Z 증가) 0.1m 구간 필터
#     mask = (H >= H10) & (H <= H10 + 0.1)
#     sel_pts = points[mask]
#     sel_cols = colors[mask]

#     # 5) Open3D로 시각화
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(sel_pts)
#     pcd.colors = o3d.utility.Vector3dVector(sel_cols)
#     # o3d.visualization.draw_geometries(
#     #     [pcd],
#     #     window_name=f"10th Nearest (Z) Window: Z10={H10:.3f}→{(H10+0.1):.3f} m"
#     # )
#     cluster_and_visualize_with_volume_filter(sel_pts,sel_cols)




# if __name__ == '__main__':
#     demo()









################################################ gpu로 바꿔봄
# import time
# from contextlib import contextmanager
# import numpy as np
# import open3d as o3d
# from scipy.spatial import ConvexHull

# # GPU DBSCAN 헬퍼
# import cupy as cp
# from cuml.cluster import DBSCAN as cuDBSCAN

# def gpu_dbscan_labels(pts: np.ndarray,
#                       eps: float,
#                       min_samples: int) -> np.ndarray:
#     """
#     pts: (N,3) numpy array
#     → GPU에서 DBSCAN 실행 후 numpy labels 반환
#     """
#     pts_gpu = cp.asarray(pts)
#     cu_db = cuDBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
#     labels_gpu = cu_db.fit_predict(pts_gpu)
#     return cp.asnumpy(labels_gpu)

# @contextmanager
# def timer(name: str):
#     start = time.perf_counter()
#     yield
#     elapsed = time.perf_counter() - start
#     print(f"[TIMER] {name}: {elapsed:.3f} s")

# def cluster_and_visualize_with_volume_filter(pts: np.ndarray,
#                                              cols: np.ndarray,
#                                              eps: float = 0.005,
#                                              min_samples: int = 10,
#                                              min_volume: float = 1e-5):
#     """
#     pts, cols로 GPU-DBSCAN → convex hull volume filter → 시각화
#     """
#     def compute_convex_hull_volume(pts: np.ndarray) -> float:
#         if pts.shape[0] < 4:
#             return 0.0
#         hull = ConvexHull(pts)
#         return hull.volume

#     # 1) GPU-accelerated DBSCAN
#     with timer("cuML DBSCAN"):
#         labels = gpu_dbscan_labels(pts, eps, min_samples)
#     max_label = labels.max()
#     print(f"[DBSCAN] Detected {max_label + 1} clusters (+ noise)")

#     # 2) 전체 원본 색상 시각화
#     with timer("전체 시각화"):
#         pcd_all = o3d.geometry.PointCloud()
#         pcd_all.points = o3d.utility.Vector3dVector(pts)
#         pcd_all.colors = o3d.utility.Vector3dVector(cols)
#         # o3d.visualization.draw_geometries([pcd_all],
#         #     window_name="All Points (Original Colors)")

#     # 3) 군집별 볼륨 계산 및 시각화
#     with timer("군집별 볼륨 계산 및 시각화"):
#         for lbl in range(max_label + 1):
#             mask = labels == lbl
#             if not np.any(mask):
#                 continue

#             cluster_pts = pts[mask]
#             vol = compute_convex_hull_volume(cluster_pts)
#             print(f" Cluster {lbl}: volume = {vol:.3e} m^3")
#             if vol < min_volume:
#                 print(f"  → Skipped (volume below {min_volume:.3e})")
#                 continue

#             pcd_cluster = o3d.geometry.PointCloud()
#             pcd_cluster.points = o3d.utility.Vector3dVector(cluster_pts)
#             pcd_cluster.colors = o3d.utility.Vector3dVector(cols[mask])
#             # o3d.visualization.draw_geometries(
#             #     [pcd_cluster],
#             #     window_name=f"Cluster {lbl} (vol={vol:.3e} m^3)"
#             # )

# def demo():
#     # 1) 포인트 클라우드 취득
#     points, colors = get_aligned_point_cloud_from_realsense()

#     # 2) '높이' H는 Z축 값
#     H = points[:, 2]

#     # 3) Z가 작은 순서로 10번째 값 찾기
#     idx_near10 = np.argpartition(H, 10)[:10]
#     tenth_idx = idx_near10[np.argmax(H[idx_near10])]
#     H10 = H[tenth_idx]

#     # 4) Z10 ~ Z10+0.1 필터
#     mask = (H >= H10) & (H <= H10 + 0.1)
#     sel_pts = points[mask]
#     sel_cols = colors[mask]

#     # 5) 기존 시각화 주석 처리
#     # pcd = o3d.geometry.PointCloud()
#     # pcd.points = o3d.utility.Vector3dVector(sel_pts)
#     # pcd.colors = o3d.utility.Vector3dVector(sel_cols)
#     # o3d.visualization.draw_geometries(
#     #     [pcd],
#     #     window_name=f"10th Nearest (Z) Window: Z10={H10:.3f}→{(H10+0.1):.3f} m"
#     # )

#     # GPU DBSCAN + Volume Filter 시각화
#     cluster_and_visualize_with_volume_filter(sel_pts, sel_cols)

# if __name__ == '__main__':
#     demo()


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
                                             min_volume: float = 1e-5):
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

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cluster_pts)
            pcd.colors = o3d.utility.Vector3dVector(cols[mask])
            o3d.visualization.draw_geometries([pcd], window_name=f"Cluster {lbl}")

def demo():
    points, colors = get_aligned_point_cloud_from_realsense()
    H = points[:, 2]
    idx_near10 = np.argpartition(H, 10)[:10]
    tenth_idx = idx_near10[np.argmax(H[idx_near10])]
    H10 = H[tenth_idx]
    mask = (H >= H10) & (H <= H10 + 0.1)
    sel_pts = points[mask]
    sel_cols = colors[mask]
    cluster_and_visualize_with_volume_filter(sel_pts, sel_cols)

if __name__ == '__main__':
    demo()