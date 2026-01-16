import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import random
import math


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


def angle_between_normals(n1, n2):
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


# def detect_large_merged_planes(points, colors=None, min_area=0.01, dist_thresh=0.015, max_planes=10, angle_thresh_deg=3.0, height_thresh=0.015):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     if colors is not None:
#         pcd.colors = o3d.utility.Vector3dVector(colors)

#     remaining = pcd
#     plane_models = []
#     plane_clouds = []

#     for _ in range(max_planes):
#         if len(remaining.points) < 100:
#             break

#         model, inliers = remaining.segment_plane(
#             distance_threshold=dist_thresh,
#             ransac_n=3,
#             num_iterations=1000
#         )
#         inlier_cloud = remaining.select_by_index(inliers)
#         if inlier_cloud.is_empty():
#             break

#         aabb = inlier_cloud.get_axis_aligned_bounding_box()
#         extent = aabb.get_extent()
#         area = extent[0] * extent[1]

#         if area >= min_area:
#             plane_models.append(model)
#             plane_clouds.append(inlier_cloud)

#         remaining = remaining.select_by_index(inliers, invert=True)

#     # 병합
#     merged_planes = []
#     used = [False] * len(plane_models)

#     for i, (n1_d1, cloud1) in enumerate(zip(plane_models, plane_clouds)):
#         if used[i]:
#             continue
#         normal1 = np.array(n1_d1[:3])
#         d1 = n1_d1[3]
#         merged = cloud1
#         used[i] = True

#         for j in range(i + 1, len(plane_models)):
#             if used[j]:
#                 continue
#             normal2 = np.array(plane_models[j][:3])
#             d2 = plane_models[j][3]

#             angle = angle_between_normals(normal1, normal2)
#             height_diff = abs(d1 - d2)

#             if angle < angle_thresh_deg and height_diff < height_thresh:
#                 merged += plane_clouds[j]
#                 used[j] = True

#         # 색상 부여 후 추가
#         merged.paint_uniform_color([random.uniform(0.2, 1.0) for _ in range(3)])
#         merged_planes.append(merged)

#     return merged_planes, remaining

def angle_between_normals(n1, n2):
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def detect_large_merged_planes(points, colors=None, min_area=0.01, dist_thresh=0.015,
                                               max_planes=10, angle_thresh_deg=3.0, height_thresh=0.015,
                                               position_thresh=0.08, cluster_eps=0.02, cluster_min_points=20):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    remaining = pcd
    plane_models = []
    plane_clouds = []

    for _ in range(max_planes):
        if len(remaining.points) < 100:
            break

        model, inliers = remaining.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=1000
        )
        inlier_cloud = remaining.select_by_index(inliers)
        if inlier_cloud.is_empty():
            break

        aabb = inlier_cloud.get_axis_aligned_bounding_box()
        extent = aabb.get_extent()
        area = extent[0] * extent[1]

        if area >= min_area:
            plane_models.append(model)
            plane_clouds.append(inlier_cloud)

        remaining = remaining.select_by_index(inliers, invert=True)

    # 병합
    merged_planes = []
    used = [False] * len(plane_models)

    for i, (n1_d1, cloud1) in enumerate(zip(plane_models, plane_clouds)):
        if used[i]:
            continue
        normal1 = np.array(n1_d1[:3])
        d1 = n1_d1[3]
        merged = cloud1
        center1 = np.mean(np.asarray(cloud1.points), axis=0)
        used[i] = True

        for j in range(i + 1, len(plane_models)):
            if used[j]:
                continue
            normal2 = np.array(plane_models[j][:3])
            d2 = plane_models[j][3]

            angle = angle_between_normals(normal1, normal2)
            height_diff = abs(d1 - d2)
            center2 = np.mean(np.asarray(plane_clouds[j].points), axis=0)
            center_dist = np.linalg.norm(center1 - center2)

            if angle < angle_thresh_deg and height_diff < height_thresh and center_dist < position_thresh:
                merged += plane_clouds[j]
                used[j] = True

        # 클러스터링하여 떨어진 부분 분리
        labels = np.array(merged.cluster_dbscan(eps=cluster_eps, min_points=cluster_min_points))
        for k in np.unique(labels):
            if k == -1:
                continue
            cluster = merged.select_by_index(np.where(labels == k)[0])
            if len(cluster.points) < 30:
                continue
            cluster.paint_uniform_color([random.uniform(0.2, 1.0) for _ in range(3)])
            merged_planes.append(cluster)

    return merged_planes, remaining
# if __name__ == "__main__":
#     print("[📷] Capturing aligned point cloud from RealSense...")
#     points, colors = get_aligned_point_cloud_from_realsense()

#     print("[📐] Detecting merged flat planes...")
#     planes, others = detect_large_merged_planes(points, colors)

#     others.paint_uniform_color([0.5, 0.5, 0.5])  # 회색

#     print("[👁️] Visualizing...")
#     o3d.visualization.draw_geometries(planes + [others])
if __name__ == "__main__":
    print("[📷] Capturing aligned point cloud from RealSense...")
    points, colors = get_aligned_point_cloud_from_realsense()

    print("[📐] Detecting merged flat planes...")
    planes, others = detect_large_merged_planes(points, colors)

    others.paint_uniform_color([0.5, 0.5, 0.5])  # 회색

    print("[✏️] Extracting plane contours...")
    contour_lines = []

    for plane in planes:
        pts = np.asarray(plane.points)
        if len(pts) < 10:
            continue

        # 2D Convex Hull을 위한 평면 상 투영
        centroid = np.mean(pts, axis=0)
        z = np.array([0, 0, 1])
        normal = plane.normals[0] if plane.has_normals() else np.array([0, 0, 1])
        if np.linalg.norm(normal) < 1e-5:
            normal = np.array([0, 0, 1])
        normal = normal / np.linalg.norm(normal)

        if abs(normal @ z) > 0.95:  # 수직 회피
            tangent = np.array([1, 0, 0])
        else:
            tangent = np.array([0, 0, 1])

        u = np.cross(normal, tangent)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        # 투영
        rel_pts = pts - centroid
        pts_2d = np.stack([rel_pts @ u, rel_pts @ v], axis=1)

        # Convex Hull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts_2d)
            contour_2d = pts_2d[hull.vertices]
            contour_3d = centroid + contour_2d[:, 0:1] * u + contour_2d[:, 1:2] * v

            # 선분 시각화용 LineSet 생성
            lines = [[i, (i + 1) % len(contour_3d)] for i in range(len(contour_3d))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(contour_3d),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.paint_uniform_color([1, 0, 0])  # 빨간색 윤곽선
            contour_lines.append(line_set)
        except:
            continue

    print("[👁️] Visualizing...")
    o3d.visualization.draw_geometries(planes + [others] + contour_lines)
