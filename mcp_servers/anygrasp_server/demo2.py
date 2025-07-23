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

    # T_open3d_to_global = np.array([
    #     [ 0, -1,  0],
    #     [-1,  0,  0],
    #     [ 0,  0, -1]
    # ])

    # # Realsense 포인트 추출 후 변환
    # points = points @ T_open3d_to_global.T
    # points = np.asarray(points, dtype=np.float32)
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

    T_open3d_to_global = np.array([
        [ 0, -1,  0],
        [-1,  0,  0],
        [ 0,  0, -1]
    ])  # 이미 정의한 변환 행렬




    best_grasp = max(filtered, key=lambda g: g.score)
    R_grasp_camera = best_grasp.rotation_matrix  # AnyGrasp 출력
    R_grasp_global = T_open3d_to_global @ R_grasp_camera
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
        o3d.visualization.draw_geometries([*grippers_all, cloud])

        # 최고 점수 하나만 다시 강조
        gripper_best = best_grasp.to_open3d_geometry()
        gripper_best.transform(trans_mat)
        print("[⭐] Showing best grasp only...")
        o3d.visualization.draw_geometries([gripper_best, cloud])


def demo():
    points, colors = get_aligned_point_cloud_from_realsense()
    analyze_point_cloud_with_anygrasp(points, colors)


if __name__ == '__main__':
    demo()
