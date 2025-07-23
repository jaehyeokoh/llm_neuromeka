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


def apply_mask_to_pointcloud(points, colors, mask_2d):
    mask_flat = mask_2d.reshape(-1).astype(bool)
    if mask_flat.shape[0] != points.shape[0]:
        raise ValueError("Mask and pointcloud size mismatch.")
    return points[mask_flat], colors[mask_flat]


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

    # # 0) 마스크 적용 (선택적)
    # if mask_ori is not None:
    #     if mask_ori.shape[0] != pts.shape[0]:
    #         raise ValueError("mask shape mismatch with pts")
    #     pts = pts[mask_ori]
    #     cols = cols[mask_ori]

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

            o3d.visualization.draw_geometries(
                to_draw,
                window_name=f"Cluster {lbl} with Contour"
            )
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

            # # (4) 기존 점군 + 옆면 점군 같이 보기
            o3d.visualization.draw_geometries(
                [pcd, pcd_side],
                window_name=f"Cluster {lbl} Side Surface"
            )


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
    o3d.visualization.draw_geometries(
        [final_visualization_pcd],
        window_name="Initial + Processed Clusters"
    )
    # 시각화 이후, final_visualization_pcd를 NumPy 배열로 변환
    final_points_np, final_colors_np = get_point_cloud_as_numpy(final_visualization_pcd)
    analyze_point_cloud_with_anygrasp(final_points_np, final_colors_np)


if __name__ == '__main__':
    demo()