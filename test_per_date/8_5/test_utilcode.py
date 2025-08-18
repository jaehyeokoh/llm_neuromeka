from typing import List, Dict

# === 방향 맵핑 그룹 ===
direction_map_groups = {
    "above": ["above", "up"],
    "below": ["below", "down"],
    "left": ["left"],
    "right": ["right"],
    "front": ["front", "forward", "straight"],
    "behind": ["back", "behind"],
    "center": ["center", "middle"]
}

def map_direction(input_word: str) -> str:
    input_word = input_word.lower()
    for standard, synonyms in direction_map_groups.items():
        if input_word in synonyms:
            return standard
    return "center"  # fallback default

# === 변환 함수 ===
def convert_plan_to_commands(parsed_plan: List[Dict[str, str]]) -> List[Dict[str, str]]:
    commands = [{"hard_agent": "assemble"}]
    
    for step in parsed_plan:
        target = step["target_object"]
        ref = step["reference_object"]
        rel_pos = step["relative_position"].strip().lower()
        mapped_direction = map_direction(rel_pos)
        dist = step.get("distance", "None")
        orient = step.get("object_orientation", "None").lower()
        
        # Move to target
        commands.append({"agent": "move", "action": "to", "target": target})
        
        # Grip
        commands.append({"agent": "grip", "action": "to", "target": target})
        
        # Tilt if needed
        if orient in ("vertical", "parallel"):
            tilt_cmd = {"agent": "tilt", "action": orient, "target": None}
            
            if orient == "parallel":
                align_ref = step.get("alignment_reference", "")
                # 문자열이면 콤마로 리스트 변환
                if isinstance(align_ref, str) and "," in align_ref:
                    align_ref_list = [r.strip() for r in align_ref.split(",")]
                else:
                    align_ref_list = align_ref  # 리스트일 경우 그대로, 단일 문자열도 그대로
                
                tilt_cmd["alignment_reference"] = align_ref_list
            
            commands.append(tilt_cmd)

        if ref == "None":
            ref = "base" # 리퍼런스가 없다면 내가 지정한 base로 설정

        # 문자열일 경우 콤마로 분리 시도
        if isinstance(ref, str) and "," in ref:
            ref_list = [r.strip() for r in ref.split(",")]
        else:
            ref_list = [ref] if isinstance(ref, str) else ref  # 리스트면 그대로, 문자열이면 1개짜리 리스트

        # 다중 타겟이 move로 들어올 경우 action에 center 넣어서 그 두 물체 사이 중심에 넣게 함
        if isinstance(ref_list, list) and len(ref_list) == 2:
            move_cmd = {"agent": "move", "action": "center_of_2_obj", "target": ref_list}
        else:
            move_cmd = {"agent": "move", "action": mapped_direction, "target": ref}

        if dist.lower() != "none":
            move_cmd["gap"] = dist # gap이 있으면 추가
        commands.append(move_cmd)
        
        # Release
        commands.append({"agent": "release", "action": "to", "target": target})
    
    # 마지막에 done 추가
    commands.append({'agent': 'done'})
    return commands

plan = [{'step': '1', 'target_object': 'black_cube_0', 'reference_object': 'None', 'relative_position': 'center', 'distance': 'None', 'object_orientation': 'vertical', 'analysis': 'Place first upright support black_cube_0 with flat top surface'}, {'step': '2', 'target_object': 'black_cube_1', 'reference_object': 'black_cube_0', 'relative_position': 'right', 'distance': 'white_box_0', 'object_orientation': 'vertical', 'analysis': 'Place second upright support black_cube_1 with flat top surface, spaced right from the first to form a gap smaller then white_box_0'}, {'step': '3', 'target_object': 'white_box_0', 'reference_object': 'black_cube_0, black_cube_1', 'relative_position': 'above', 'distance': 'None', 'object_orientation': 'parallel', 'analysis': 'Place a wide, flat capstone white_box_0 horizontally across both supports', 'alignment_reference': 'black_cube_0, black_cube_1'}]
plan2 = [{'step': '1', 'target_object': 'black_cube_0', 'reference_object': 'None', 'relative_position': 'center', 'distance': 'None', 'object_orientation': 'vertical', 'analysis': 'Place first upright support black_cube_0 with flat top surface'}, {'step': '2', 'target_object': 'black_cube_1', 'reference_object': 'black_cube_0', 'relative_position': 'right', 'distance': 'white_box_0', 'object_orientation': 'vertical', 'analysis': 'Place second upright support black_cube_1 with flat top surface, spaced right from the first to form a gap smaller then white_box_0'}, {'step': '3', 'target_object': 'white_box_0', 'reference_object': 'black_cube_0, black_cube_1', 'relative_position': 'above', 'distance': 'None', 'object_orientation': 'parallel', 'analysis': 'Place a wide, flat capstone white_box_0 horizontally across both supports', 'alignment_reference': 'black_cube_0, black_cube_1'}]
converted = convert_plan_to_commands(plan)
# converted.append({'agent': 'done'})
print(converted)
print(converted[3])

action_num = 0
user_input_analyzed_by_LLM = converted
if isinstance(user_input_analyzed_by_LLM, list) and user_input_analyzed_by_LLM:
    if "hard_agent" in user_input_analyzed_by_LLM[0]:
        print(f"hard_agent 키 발견, {user_input_analyzed_by_LLM[0]}코드 실행")
        bias = 1

# 액션 다 끝나면 종료 및 action num 초기화
print(len(user_input_analyzed_by_LLM))




from itertools import zip_longest

# === 타입 통합 처리 ===
if isinstance(user_input_analyzed_by_LLM, dict):
    parsed_list = list(user_input_analyzed_by_LLM.values())
elif isinstance(user_input_analyzed_by_LLM, list):
    parsed_list = user_input_analyzed_by_LLM
else:
    raise TypeError("user_input_analyzed_by_LLM must be a list or dict.")

# === bias 적용 및 zip_longest으로 안전하게 curr/next 처리 ===
for parsed, parsed_next in zip_longest(parsed_list[bias:], parsed_list[bias+1:]):
    if parsed["agent"] == "move":
        if parsed_next and parsed_next["agent"] == "grip":
            print("move and grip")
            print(parsed)
            if "gap" in parsed:
                print("gap is ",parsed["gap"])
        elif parsed_next and parsed_next["agent"] == "release":
            print("move before release")
            print(parsed)
            if "gap" in parsed:
                print("gap is ",parsed["gap"])
        else:
            print("only move")
            print(parsed)
            if "gap" in parsed:
                print("gap is ",parsed["gap"])

    elif parsed["agent"] == "grip":
        print("gripper")
        print(parsed)

    elif parsed["agent"] == "release":
        print("release")
        print(parsed)


    elif parsed["agent"] == "tilt":
        print("tilt")
        print(parsed)
        if "alignment_reference" in parsed:
            print("alignment reference is ", parsed["alignment_reference"][0],parsed["alignment_reference"][1] )

    elif parsed["agent"] == "done":
        print("done")


list1 = [1,2,3,4,5,6]
print(list1[:3])