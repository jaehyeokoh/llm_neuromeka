import argparse
import torch
import numpy as np
import warnings
from MathFunc import MathFunc
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

##################################################################3
"""
AnyGrasp import시 UTF-8 오류 완전 차단
"""
import os, sys, io

# 환경변수 설정
os.environ.update({
    'LC_ALL': 'C',
    'LC_TIME': 'C', 
    'LANG': 'C',
    'PYTHONIOENCODING': 'utf-8'
})

# 파일 디스크립터 레벨 차단 (anygrasp에서 자꾸 utf-8이 아닌 문자를 출력해서 mcp서버가 멈추는 버그 해결하기 위해 이와같은 로깅 차단 코드 넣음)
stdout_fd = os.dup(1)
stderr_fd = os.dup(2) 
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 1)
os.dup2(devnull, 2)

# Python 레벨 차단
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

try:
    from gsnet import AnyGrasp
finally:
    # 복구
    os.dup2(stdout_fd, 1)
    os.dup2(stderr_fd, 2)
    os.close(stdout_fd)
    os.close(stderr_fd)
    os.close(devnull)
    sys.stdout = original_stdout
    sys.stderr = original_stderr

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
###################################################################




from graspnetAPI import GraspGroup
from mcp.server.fastmcp import FastMCP, Context
import base64, json, logging, gc, builtins
from contextlib import contextmanager, asynccontextmanager
from typing import AsyncIterator
from dataclasses import dataclass

os.environ['PYTHONIOENCODING'] = 'utf-8'

# 모든 print 비활성화
original_print = builtins.print
def safe_print(*args, **kwargs):
    try:
        original_print(*[str(arg) for arg in args], **kwargs)
    except:
        pass
builtins.print = safe_print

# 루트 로거에 달려 있는 기존 핸들러 모두 제거
logging.getLogger().handlers.clear()

# 전파 방지 (중복 로깅 방지)
logging.getLogger().propagate = False

# 기본 로그 설정 복구
# logging.basicConfig(level=logging.WARNING)  # WARNING 이상 로그만 출력
logging.disable(logging.CRITICAL)



# GPU 메모리 가드
@contextmanager  
def gpu_memory_guard():
    """GPU 메모리 안전 관리"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# 컨텍스트 클래스
@dataclass
class AnyGraspContext:
    model: 'AnyGrasp'

# Lifespan 관리
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AnyGraspContext]:
    """서버 시작 시 AnyGrasp 모델 로드, 종료 시 GPU 메모리 해제"""
    
    model = None
    try:
        # AnyGrasp 모델 로드 (기존과 동일)
        model = AnyGrasp(cfgs)
        model.load_net()
        model.net.eval()

        # 컨텍스트 전달
        yield AnyGraspContext(model=model)

    finally:
        
        # 모델 정리
        if model is not None:
            try:
                del model
            except:
                pass
        
        # GPU 메모리 완전 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 가비지 컬렉션
        gc.collect()

# 서버에 고유한 이름 부여
mcp = FastMCP("anygrasp_mcp_server", lifespan=app_lifespan)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=False, default="checkpoint/checkpoint_detection.tar", help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.67, help='Maximum gripper width (<=0.1m)') # 그리퍼의 최대 폭 (작으면 작을 수록 물체에 생기는 grasp point줄어들고 크면 못잡는걸 잡는다고 나옴)
parser.add_argument('--gripper_height', type=float, default=0.029, help='Gripper height')  # 이거랑 width가 collision detection에 영향 주는듯
parser.add_argument('--top_down_grasp', action='store_true', default=False, help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', default=True, help='Enable debug mode')
parser.add_argument('--collision_detection', default=True, help='Enable collision_detection mode')
cfgs = parser.parse_args()

def decode_point_cloud(data: dict):
    points = np.frombuffer(base64.b64decode(data["points"]), dtype=np.float32).reshape(-1, 3)
    colors = np.frombuffer(base64.b64decode(data["colors"]), dtype=np.float32).reshape(-1, 3)
    return points, colors

# def move_to_grasp(self, pos_cam_to_gripper_in_cam, rot_cam_to_gripper):
def grasp_rpy( rot_cam_to_gripper):
        

        robot_end_ori = MathFunc.degree_to_rad(np.array([180.0,0.0,180.0]))
        rot_BE = MathFunc.euler_to_rotMat(
            euler_x=robot_end_ori[0], euler_y=robot_end_ori[1], euler_z=robot_end_ori[2])
        

        rot_EC = np.array( # End-effector → Camera의 회전행렬
            [[0., 1., 0.],
            [-1., 0., 0.],
            [ 0., 0., 1.]], dtype=np.float32)
 

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
 
        # Solve for Frame: fixed E (***)
        rot_BGfixedE = rot_BG @ rot_GE_fixed
        #####
        # Project 6D grasp to 4D grasp (translation + yaw)
        euler_xyz = MathFunc.rotMat_to_euler(rot_BGfixedE)
        rpy = np.rad2deg(euler_xyz)
        #####

        return [rpy[0],rpy[1],rpy[2]]




@mcp.tool()
async def anygrasp_analyze(data, ctx: Context, target_boundary_info=None, debug=False):
    """
    data : 포인트 클라우드
    target_boundary_info : 타겟의 윗면 영역, 높이
    debug : 키면 시각정보데이터 (그리퍼를 포인트 클라우드에 추가한다던가) 활성화  -> 근데 서버간 데이터 통신 커지면 렉걸리니 일단 껐음

    함수 목적 : 포인트 클라우드(GroundedSAM2로 객체 옆면 보간한)를 통해 파지점과 방향 추출
    """
    cfgs.debug = debug
    
    # 컨텍스트에서 모델 가져오기
    app_context: AnyGraspContext = ctx.request_context.lifespan_context
    ANYGRASP_MODEL = app_context.model
    
    # 모든 변수를 미리 None으로 초기화
    colors = None
    gg = None
    cloud = None
    filtered_gg = None
    grippers_all = None

    # 전체 함수를 GPU 메모리 가드로 감싸기
    with gpu_memory_guard(): # -> 이건 try - except -> finally대신 사용하는거임 with 내부의 블록이 끝나든 강제종료 되든 무조건 gpu_memory_guard를 실행 후 종료됨, gpu_memory_guard는 gc 및 cuda변수 삭제 코드임
        try:
            points, colors = decode_point_cloud(data)
            
            lims = [-1.765, 1.717,  # xmin, xmax # 8_6 limit 수정됨 -> 절대좌표로 해서
                    -1.779, 1.777,  # ymin, ymax  
                    -1.34,  1.31]  # zmin, zmax]

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
                # print('No Grasp detected after collision detection!')
                return "error 1"
                # pass
            
            gg = gg.nms().sort_by_score()

            #  target_side_pts 기반 필터링 (target의 범위 내에 있는 포인트만 남기는 코드)
            if target_boundary_info != None:
                try:
                    import pickle
                    boundary_info = pickle.loads(base64.b64decode(target_boundary_info))
                    
                    # 문자열을 float로 변환
                    hull_vertices = [[float(x), float(y)] for x, y in boundary_info["hull_vertices"]]
                    # z_range = boundary_info["z_range"]
                    
                    def point_in_polygon(point, polygon, offset=0.02):
                        x, y = point
                        
                        # 오프셋 적용
                        if offset > 0:
                            min_x = min(p[0] for p in polygon) - offset
                            max_x = max(p[0] for p in polygon) + offset
                            min_y = min(p[1] for p in polygon) - offset
                            max_y = max(p[1] for p in polygon) + offset
                            
                            if min_x <= x <= max_x and min_y <= y <= max_y:
                                return True
                        
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
                        
                        # if (z_range["min"]-1.3 <= z <= z_range["max"]+1.3 and 
                        #     point_in_polygon([x, y], hull_vertices)): # z범위 지정하려고 했는데 이상하게 나옴
                        #     filtered_gg.append(g)
                        if point_in_polygon([x, y], hull_vertices): # z범위 지정하려고 했는데 이상하게 나와서 z범위를 없앰
                            filtered_gg.append(g)
                    
                    gg = filtered_gg if len(filtered_gg) > 0 else gg
                    
                except Exception as e:
                    # pass  # 실패시 기존 grasp 유지
                    return f"error {e}"
            
            if len(gg) == 0:
                return "error 2"
            
            json_str = None
            # 시각화
            if debug:
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
        
        finally:
            # 안전한 메모리 정리
            try:
                # 변수들 정리
                for var in [points, colors, gg, cloud, filtered_gg, grippers_all]:
                    if var is not None:
                        del var
                
                # 가비지 컬렉션
                gc.collect()
                
            except Exception as cleanup_error:
                pass  # 정리 중 오류는 무시


# # --- MCP 서버 시작 ---
if __name__ == "__main__":
    print("MCP 서버 루프 시작...")
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\n사용자에 의해 서버 중지됨 (Ctrl+C).")
    except Exception as e:
        print(f"\nMCP 서버 오류 발생: {e}")

