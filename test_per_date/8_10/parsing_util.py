"""Parsing utilities extracted from utils.py

Behavior preserved. Two functions use small shared helpers (below):
- parse_llm_non_json2json_output_for_vlm
- parse_llm_non_json2json_output_for_input_matching
"""
import re
import ast
import json

# ------------------------ Common parsing helpers (Option A) ------------------------
def _parse_numbered_steps(lines):
    """Return lines like '1. ...', preserving order."""
    return [ln for ln in lines if re.match(r"^\d+\.\s", ln)]

def _after_answer(lines):
    """Return lines after the first 'answer:' (case-insensitive)."""
    started = False
    out = []
    for ln in lines:
        if not started:
            if ln.strip().lower().startswith("answer:"):
                started = True
            continue
        if ln.strip():
            out.append(ln.strip())
    return out

def _parse_action_objects_line(s: str):
    """Parse 'action -- obj1, obj2' -> (action, [objs])."""
    if "--" in s:
        action_part, objects_part = s.split("--", 1)
        action = action_part.strip()
        objs = [o.strip() for o in objects_part.split(",") if o.strip()]
    else:
        action = s.strip()
        objs = []
    return action, objs

def _normalize_text(s: str) -> str:
    """Normalize line endings and strip top-level code fences; whitespace-trim."""
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"^\s*```.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*```$", "", s, flags=re.MULTILINE)
    return s

def _split_reasoning_answer(s: str, *, missing_sep_msg: str = "Output missing '###' separator"):
    """Return (reasoning_lines, answer_lines); raise with caller's message if missing '###'."""
    s = _normalize_text(s)
    try:
        reasoning_block, answer_block = s.split("###")
    except ValueError:
        raise ValueError(missing_sep_msg)
    reasoning_lines = [ln.strip() for ln in reasoning_block.splitlines() if ln.strip()]
    answer_lines = [ln.strip() for ln in answer_block.splitlines() if ln.strip()]
    return reasoning_lines, answer_lines

def _parse_answer_list(line: str, *, missing_prefix_msg: str = "Output block must start with 'answer:'"):
    """Parse 'answer: a, b' or 'answer: None' -> list[str]."""
    if not line.lower().startswith("answer:"):
        raise ValueError(missing_prefix_msg)
    content = line[len("answer:"):].strip()
    if content.lower() == "none":
        return []
    return [x.strip() for x in content.split(",") if x.strip()]

def _parse_kv_lists(lines):
    """Parse lines like 'matched_ids: [...]' and 'missing: [...]' using ast.literal_eval."""
    out = {}
    for ln in lines:
        m = re.match(r"(matched_ids|missing)\s*:\s*(.+)", ln)
        if m:
            key, raw = m.group(1), m.group(2)
            try:
                val = ast.literal_eval(raw)
                if isinstance(val, list):
                    out[key] = val
            except Exception as e:
                raise ValueError(f"Could not parse list value for '{key}': {raw}") from e
    return out

def _prepare_json_string(s: str, *, allow_single_quotes: bool = False, strip_np_int64: bool = False) -> str:
    """Return a JSON-like string after optional normalizations.
    - allow_single_quotes: replace single quotes with double quotes
    - strip_np_int64: remove "np.int64(" ... ")" wrappers (flat replace)
    """
    if s is None:
        return s
    t = s
    if strip_np_int64:
        t = t.replace("np.int64(", "").replace(")", "")
    if allow_single_quotes:
        t = t.replace("'", '"')
    return t
# ----------------------------------------------------------------------------------

def extract_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    # 리스트 혹은 객체 시작 위치 탐색
    start = raw.find("{")
    start_list = raw.find("[")
    if (start_list != -1 and (start_list < start or start == -1)):
        start = start_list

    # 끝 위치 탐색
    end_obj = raw.rfind("}")
    end_list = raw.rfind("]")
    end = max(end_obj, end_list)

    if start == -1 or end == -1:
        raise ValueError("JSON 객체 또는 리스트를 찾을 수 없습니다.")

    # 추출 및 파싱
    json_str = raw[start:end+1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 실패: {e}")

    return data


def convert_list_to_dict(obj_list):
    result = {}
    for obj in obj_list:
        name = obj.get("name")
        if name:
            result[name] = obj
    return result


def format_actions_for_llm(action_list):
    formatted = []
    for action in action_list:
        if isinstance(action, dict):
            act = action.get("action")
            objs = action.get("objects", [])
            if act:
                if objs:
                    formatted.append(f"{act} -- {', '.join(objs)}")
                else:
                    formatted.append(f"{act}")
        elif isinstance(action, str):
            formatted.append(action)
    return formatted


def parse_plan_editor_output(text: str):
    """
    LLM divided_input editor 출력 파싱 함수.
    - reasoning: Step-by-step 문자열 리스트
    - steps: [{"action": "...", "objects": ["..."]}] 형태의 리스트
    """
    result = {"reasoning": [], "steps": []}
    parts = text.strip().split("###")
    if len(parts) != 2:
        print("[parse_plan_editor_output] error occured :", text)
        raise ValueError("Output missing '###' separator")

    # reasoning
    for line in parts[0].splitlines():
        if line.strip().lower().startswith("step"):
            result["reasoning"].append(line.strip())

    # steps
    current_section = None
    for raw in parts[1].splitlines():
        s = raw.strip()
        if not s:
            continue
        low = s.lower()
        if low == "answer:":
            current_section = "answer"
            continue
        if low == "steps:":
            current_section = "steps"
            continue
        if current_section == "steps":
            m = re.match(r"^\d+\.\s*(.+)$", s)
            if m:
                action, objs = _parse_action_objects_line(m.group(1).strip())
                result["steps"].append({"action": action, "objects": objs})
    return result


def parse_llm_non_json2json_output_for_vlm(text: str) -> dict:
    """
    Parses LLM output of the form:
    
    reasoning:
    <multiple lines>
    ###
    answer:
    <comma-separated list of objects> or "None"
    """
    # 강제 문자열 분리
    reasoning_lines, answer_lines = _split_reasoning_answer(
        text, missing_sep_msg="Output missing '###' separator"
    )

    # reasoning 축약 저장(비어있지 않은 라인만)
    reasoning = [ln for ln in reasoning_lines if ln]

    # answer 라인 검증 및 파싱
    if not answer_lines:
        raise ValueError("Output block must start with 'answer:'")
    selected_line = None
    for ln in answer_lines:
        if ln.lower().startswith("answer:"):
            selected_line = ln
            break
    if selected_line is None:
        raise ValueError("Output block must start with 'answer:'")

    objects = _parse_answer_list(selected_line, missing_prefix_msg="Output block must start with 'answer:'")
    return {"reasoning": reasoning, "answer": objects}


def parse_llm_non_json2json_output_for_input_matching(text: str) -> dict:
    """
    Parses LLM output for input matching of the form:

    reasoning:
    <multiple lines>
    ###
    answer:
    matched_ids: [..]
    missing: [..]
    """
    # 구간 분리
    reasoning_lines, answer_lines = _split_reasoning_answer(
        text, missing_sep_msg="Output missing '###' separator"
    )

    # reasoning 수집
    reasoning = [ln for ln in reasoning_lines if ln]

    # answer 블록은 'answer:'로 시작해야 함
    if not answer_lines or not any(ln.lower().startswith("answer:") for ln in answer_lines):
        raise ValueError("Output block must start with 'answer:'")

    # 'answer:' 이후 내용만 처리
    content_lines = _after_answer(answer_lines)
    output_dict = _parse_kv_lists(content_lines)
    return {"reasoning": reasoning, "answer": output_dict}


def parse_simple_plan_editor_output(text: str) -> dict:
    """
    Parses output of Plan Editor Agent in the following hybrid format:

    reasoning:
    <step-by-step lines>
    ###
    answer:
    missing: ["object1", "object2"]

    또는

    answer:
    1. Step one...
    2. Step two...
    """
    reasoning_lines, answer_lines = _split_reasoning_answer(
        text, missing_sep_msg="Output missing '###' separator"
    )

    # Case A: key-value lists
    output_dict = _parse_kv_lists(answer_lines)
    if output_dict:
        return {"reasoning": reasoning_lines, "answer": output_dict}

    # Case B: numbered steps
    steps = _parse_numbered_steps(answer_lines)
    return {"reasoning": reasoning_lines, "answer": steps}


def split_plan_by_grip_release(lines):
    """
    lines: plan editor 'answer' 영역의 줄 리스트
    'Grip' 또는 'Release'를 기준으로 블록을 분할함.
    각 블록은 'action -- obj1, obj2'와 같은 라인으로 시작되고,
    이어지는 상세라인(들)이 따라붙을 수 있음.
    """
    blocks = []
    current_block = []
    inside_action = False
    pending_pre_grip = []

    def flush_pending():
        nonlocal pending_pre_grip
        if pending_pre_grip:
            blocks.append(pending_pre_grip)
            pending_pre_grip = []

    for line in lines:
        # Grip/Release로 시작하는 줄은 새로운 블록의 시작
        if re.match(r"^\s*(Grip|Release)\b", line, flags=re.I):
            # 이전 블록 정리
            if current_block:
                blocks.append(current_block)
                current_block = []
                inside_action = False

            # 저장되지 않은 pre-grip 라인들 처리
            flush_pending()

            # 새 블록 시작
            current_block = [line]
            inside_action = True

        # "action -- obj1, obj2" 꼴이면, 새로운 블록으로 간주
        elif re.match(r"^\s*\w+\s*--\s*.+", line):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
            inside_action = True

        # 일반 라인
        else:
            if inside_action:
                current_block.append(line)
            else:
                pending_pre_grip.append(line)

    # 마지막 블록 누락 방지
    if current_block:
        blocks.append(current_block)

    return blocks


######################################################### Hard agent 파서
def parse_agent_assignment_output(text: str) -> dict:
    """
    Hard Agent의 assign 프롬포트 출력을 파싱함
    Parses LLM output of the form:
    reasoning:

    <lines>

    ###
    answer:
    <agent_name>
    """
    # normalize
    s = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # split by ###
    parts = s.split("###")
    if len(parts) != 2:
        raise ValueError("Output missing '###' separator")

    # reasoning
    reasoning_block = parts[0].strip()
    reasoning_lines = [ln.strip() for ln in reasoning_block.splitlines() if ln.strip()]

    # answer block
    answer_block = parts[1]
    lines = [ln for ln in answer_block.splitlines() if ln.strip()]

    # find agent name
    agent_name = None
    for line in lines:
        if line.strip().lower().startswith("answer:"):
            continue
        cleaned = line.strip()
        if cleaned:
            agent_name = cleaned
            break

    if not agent_name:
        raise ValueError("No agent name found in answer block")

    return {
        "reasoning": reasoning_lines,
        "answer": agent_name
    }


def clean_keep_outer_braces(text: str) -> str:
    """
    백틱 코드 블록(```json ... ```) 내의 JSON에서, 
    최외곽 중괄호는 유지하면서 내부에서 불필요한 들여쓰기/공백 라인을 정리한다.
    """
    # 코드 블록 제거 (json 명시 여부 무관)
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:-1])

    # 최외곽 중괄호 범위 추출
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        return text

    inner = s[start+1:end]

    # 줄 단위 정리: 앞뒤 공백/빈줄 제거
    inner_lines = [ln.rstrip() for ln in inner.splitlines()]
    # 불필요한 빈 줄 2개 초과 제거
    cleaned = []
    blank_run = 0
    for ln in inner_lines:
        if not ln.strip():
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(ln.strip())

    # 재조합
    return s[:start+1] + "\n" + "\n".join(cleaned) + "\n" + s[end:]


def ensure_list_of_dicts(raw):
    """
    raw가 문자열이면 JSON으로 파싱하고, 리스트라면 그대로 사용.
    각 원소가 dict인지 확인하고 아니면 건너뜀.
    """
    if isinstance(raw, str):
        try:
            data = json.loads(_prepare_json_string(raw))
        except json.JSONDecodeError:
            data = []
    elif isinstance(raw, list):
        data = raw
    else:
        data = []

    return [x for x in data if isinstance(x, dict)]


def parse_to_obj2(raw):
    """
    raw가 문자열이면 JSON 블록(객체/리스트) 추출·파싱.
    이미 파이썬 객체(list/dict)이면 그대로 반환.
    실패 시 빈 리스트/객체 반환(기존 관례 유지).
    """
    if isinstance(raw, (list, dict)):  
        return raw
    if raw is None:
        return {}
    if not isinstance(raw, str):
        try: raw = str(raw)
        except Exception: return {}
    s = raw.strip()
    if not s: return {}
    try:
        return extract_json(s)
    except Exception:
        if s.startswith("[") and s.endswith("]"):
            try: return json.loads(_prepare_json_string(s))
            except Exception: return []
        if s.startswith("{") and s.endswith("}"):
            try: return json.loads(_prepare_json_string(s))
            except Exception: return {}
        return {}


def parse_eval_refine_output_loose(text, keywords):
    """
    '###' 구분이 없더라도, 지정한 키워드 구간을 느슨하게 파싱한다.
    - keywords: ["reasoning", "answer"] 같은 순서 리스트
    """
    s = _normalize_text(text)
    
    flags = re.I | re.S
    result = {}
    
    for i, keyword in enumerate(keywords):
        if i == len(keywords) - 1:  # 마지막 키워드
            pattern = rf"{keyword}:\s*(?P<content>.*)\Z"
        else:
            next_keyword = keywords[i + 1]
            pattern = rf"{keyword}:\s*(?P<content>.*?)(?:\n\s*###\s*\n\s*)?{next_keyword}:"
        m = re.search(pattern, s, flags)
        if m:
            block = m.group("content").strip()
            # 라인 단위로 정리
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            result[keyword.lower()] = lines
        else:
            result[keyword.lower()] = []
    
    return result


def formatted_plan_parser(text: str) -> dict:
    """
    특정 포맷의 plan 텍스트를 파싱해서 구조화한다.
    예:
      Step 1: move -- cup, table
      Step 2: grip -- cup
    """
    result = {"steps": []}
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^Step\s+\d+:\s*(.+)$", s, flags=re.I)
        if m:
            action, objs = _parse_action_objects_line(m.group(1))
            result["steps"].append({"action": action, "objects": objs})
    return result


def convert_plan_to_commands(parsed_plan):
    """
    parsed_plan = {
      "steps": [
        {"action": "...", "objects": [...]},
        ...
      ]
    }
    형태를 받아서 내부 명령 리스트로 변환
    """
    commands = []
    for step in parsed_plan.get("steps", []):
        action = step.get("action")
        objs = step.get("objects", [])
        if action:
            commands.append({"action": action, "targets": objs})
    return commands


def parse_result_list(mixed):
    """
    문자열/리스트 혼합 입력을 표준 리스트로 변환.
    - 문자열: 줄바꿈/쉼표로 분할
    - 리스트: 문자열 요소만 채택
    """
    if isinstance(mixed, list):
        return [str(x).strip() for x in mixed if isinstance(x, (str, int, float))]
    if isinstance(mixed, str):
        tmp = []
        for part in mixed.replace(",", "\n").splitlines():
            s = part.strip()
            if s:
                tmp.append(s)
        return tmp
    return []


def parse_to_dict_for_dino(raw_string: str) -> dict:
    """
    DINO 출력 문자열을 dict로 파싱한다.
    
    처리 단계:
    1) 작은따옴표(')를 큰따옴표(")로 일괄 치환
    2) 'np.int64('와 닫는 ')' 제거
    3) json.loads로 파싱 시도
    실패 시 {} 반환
    Args:
        raw_string (str): 처리할 원본 문자열.
    
    Returns:
        dict: 파싱된 딕셔너리.
    """
    cleaned_str = _prepare_json_string(raw_string, allow_single_quotes=True, strip_np_int64=True)
    try:
        data = json.loads(cleaned_str)
        return data
    except json.JSONDecodeError as e:
        print(f"JSON 디코딩 오류: {e}")
        return {}
