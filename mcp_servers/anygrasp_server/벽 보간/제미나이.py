import numpy as np

def get_approach_vector(rotation_matrix):
    """
    주어진 3x3 회전 행렬에서 그리퍼의 접근 방향 벡터를 추출합니다.
    
    가정:
    1. 그리퍼의 로컬 X축이 접근 방향입니다. (회전 행렬의 첫 번째 열)
    2. 전역 Z축의 양의 방향은 '아래쪽'을 의미합니다.
    
    Args:
        rotation_matrix (np.ndarray): 3x3 형태의 회전 행렬.
                                    [[Rx_x, Ry_x, Rz_x],
                                     [Rx_y, Ry_y, Rz_y],
                                     [Rx_z, Ry_z, Rz_z]]
                                    여기서 R_x, R_y, R_z는 그리퍼의 로컬 X, Y, Z축이
                                    전역 좌표계에서 가리키는 방향 벡터입니다.

    Returns:
        np.ndarray: 그리퍼의 접근 방향을 나타내는 3D 벡터 (정규화됨).
                    [x_direction, y_direction, z_direction]
    """
    
    # 가정 1: 그리퍼의 로컬 X축이 접근 방향입니다.
    # 이는 회전 행렬의 첫 번째 열에 해당합니다.
    approach_vector = rotation_matrix[:, 0]
    
    # 벡터의 크기를 1로 정규화합니다.
    # 방향만 중요하므로 크기를 1로 만들면 해석하기 더 용이합니다.
    if np.linalg.norm(approach_vector) == 0:
        return np.array([0., 0., 0.]) # 0 벡터인 경우
    
    normalized_approach_vector = approach_vector / np.linalg.norm(approach_vector)
    
    # 참고: 만약 전역 Z축이 '위쪽'을 의미한다면, 
    # normalized_approach_vector[2]의 부호를 반전시켜야 합니다.
    # 하지만 현재 가정은 전역 Z축 양의 방향이 '아래쪽'이므로 그대로 사용합니다.
    
    return normalized_approach_vector

# 제공된 회전 행렬 예시
rotation_matrix_example = np.array([
    [-0.91952193,  0.24544388,  0.3069801],
    [-0.23281257, -0.96941084,  0.0777239],
    [ 0.31666666,  0.        ,  0.94853693]
])

# 접근 방향 벡터 계산
approach_direction = get_approach_vector(rotation_matrix_example)

print("회전 행렬:\n", rotation_matrix_example)
print("\n계산된 접근 방향 벡터:", approach_direction)

# 해석:
# [X_dir, Y_dir, Z_dir]
# X_dir (음수): 오른쪽에서 왼쪽으로 이동
# Y_dir (음수): 뒤쪽으로 살짝 이동
# Z_dir (양수): 전역 Z축이 아래 방향이므로, 아래쪽으로 살짝 기울어짐
# 이 벡터는 '오른쪽에서 왼쪽으로 거의 수평하게 위에서 아래로 살짝 기울어지게 접근'한다는
# 사용자님의 시각화 설명과 매우 잘 일치합니다.



# 이거 x,z,y순서임 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ