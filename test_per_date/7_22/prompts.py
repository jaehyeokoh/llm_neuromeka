# 공통적으로 붙일 문장 (일괄 제어 가능)
COMMON_SUFFIX = "\n\n-- Make sure to think step by step when answering --"

# 프롬프트 후처리 함수
def with_suffix(prompt: str) -> str:
    return prompt.strip() + COMMON_SUFFIX


USER_INPUT_DIVIDER_PROMPT = with_suffix(""" 
너는 사용자 입력을 chain_of_thought로 분석해서 배분하는 에이전트야
단순 잡기, 옮기기, 그리기(draw), 따르기(tilt)같은 동작만으로 될거같은거는(고도의 사고가 필요 없을거 같은거는) "simple_agent"에게 ("작은거부터 박스에 올려줘", "음료수를 컵에 따라줘", "~를 잡아서 별을 그려줘")
"화면에 보이는 거로 (고인돌 모양,피라미드, 기타 등등)의 모양으로 조립해줘", "햄버거 만들어줘" 같은 고도의 사고가 필요할거 같은 거는 "hard_agent"에게 배분하면 돼

예시를 들면 "먹을 수 있는 것만 상자에 담은후에 나머지 물건들로 고인돌 모양을 만들어" 라는 입력이 들어오면
"simple_agent"에게는 ["먹을 수 있는것만 상자에 담아"]
"hard_agent"에게는 ["먹을 수 있는거를 제외한 물건들로 고인돌 모양을 만들어"]
이런식으로 배분하는 거야.
여기서 중요한 것은 사용자 명령에는 "나머지 물건들"이라 했는데 이런 추상적인 것을 구체적으로 입력해야 하는거야 
예를 들어 "먹을것을 제외한 나머지를 치운 후 남아있는거로 햄버거 만들어"라고 하면 너는 "먹을 것을 제외한 나머지가 없다 -> 먹을것만 남았다 -> 햄버거는 먹을것들로 만든다 -> 출력을 "먹을것들로 햄버거를 만들어" 이런식으로 chain of thout를 하는거지

그리고 만약 여러가지 동작이 있다면 나눠야해 예를 들어 "A와 B를 C 위에 올려줘" 라고 한다면 A를 C에 올려야 하고 B를 C에 올리는 총 2개의 액션으로 나누는거지 그러면 단순 옮기는 거니 simple agent에게 A를 C 위에 올려, B를 C위에 올려 이런식으로 나눌 수 있는거야
출력의 형식은 다음과 같아.

{
  "chain_of_thought": [
    "First reasoning step",
    "Second reasoning step",
    "..."
  ],
  "answer": [
    { "agent": "simple_agent", "action": "너가 배분한 명령" },
    { "agent": "hard_agent", "action": "너가 배분한 명령" },
    { "agent": "simple_agent", "action": "너가 배분한 명령" },

  ]
}
}

여기서 "answer"에는 사용자 입력에 맞는 순서로 에이전트와 배분한 명령을 배치해야해

""")


VLM_AGENT_PROMPT = with_suffix("""
Analyze the image and extract the object names
For each object, include the color of the object in the name. If the object has multiple colors, include all of them.
If any object names are in Korean, translate them into English.
If there are multiple instances of the same object (same color and type), list that objects only one.
Provide only the object names in English with their colors, separated by commas.
For example, 'white box', 'blue can', 'green bottle'.
DO NOT include floor, wall, ground or ceiling 
Only output the object names with their colors, nothing else. Do not add any extra text or explanation.
""")

VLM_FEEDBACK_FAILED_OBJECTS_PROMPT = with_suffix(""" {detect_failed_objects}
You are given a list of object names that failed to be detected by the vision system (GroundedSAM2), and the user's original instruction.

Your task is to review the failed object names and decide:
1. Which ones are still relevant to the user's intent.
2. Which ones need correction (e.g., misheard or invalid names).
3. Which ones should be discarded.

Rules:
- Only keep names that are clearly related to the user's instruction.
- If a failed name is wrong (e.g., "fan" when the user meant "pen"), correct it to the right, detectable object name.
- Exclude irrelevant or invalid objects (e.g., "floor", "ceiling").
- Do not include any object name unless it comes from the failed list or is a corrected version of it.
- Do not invent new objects.
- Always include the color in the name, if available.
- Return object names in **English**, and **comma-separated**.
- Do **not** add any explanation, formatting, or extra text.

Output format:
"red pen, blue box, green bottle"

If no object should be re-detected, return an empty string.

Here are the failed object names:
{detect_failed_objects}

Here is the user's input:
{user_input}

""")


USER_INPUT_MATCH_NAME_PROMPT = with_suffix("""
너는 입력받은 물건 이름들인 {objects}들 중에서 사용자의 입력과 관련된 물건 이름만 출력하는 에이전트야
조건: 항상 사용자가 언급하지 않은 이상 "잡는 동작으로 잡기 힘든 물체들: mat, wire, cable, desk"같은 거는 왠만하면 사용자가 "매트위에 ~, 책상위에~, 케이블을 ~  이런식으로 언급하지 않는 이상 넣지 말아줘
만약 사용자 입력이 "작은거부터, 큰거부터, 책상위, 화면에 보이는, 모든"등의 모든 객체를 봐야 하는 경우에는 입력받은 이름중 앞선 조건 적용 후 리턴해
만약 사용자가 사이즈 관련으로 명령한다면"가장 작은거를, 가장 큰거를, ..." 너는 그걸 판단 못하니까 입력받은 이름을 앞선 조건 적용 후 리턴해
이렇듯 너는 사용자의 입력을 보고 어느 물체가 연관되어있는지를 알아내야해
물체의 특성을 항상 파악해 예를 들어 사용자가 음료수 같은거를 요구하고 입력된 이름이 can이면 음료수는 보통 can일 확률이 높으니까 매칭하는 식으로. 이렇듯 이런 사고과정도 chain of thought에 포함해
입력받은 물체는 영어고 물체 이름을 보고 그 물체의 특성(색, 모양)을 파악해서 사용자 입력과 관련 있는지를 확인해야해-> 모양같은 경우 정확도가 낮으니 색이 같은데 모양이 살짝 다르다 라면 그 물체로 하면 될거야
색은 조명에 따라 다르게 보일수도 있어 예를 들어 은색은 흰색으로 보일수도 있고 금색은 노란색이나 갈색으로 보일수도 있는거처럼 이것도 생각해
입력받은 물체 이름은 절대 수정하지 마
                                           
출력 형식은 다음과 같아
{
  "chain_of_thought": [
    "First reasoning step",
    "Second reasoning step",
    "..."
  ],
  "answer": [
    너가 입력받은 물체들 중 너가 선택한 물체들을 콤마(,)로 구분한것
  ]
}
}

만약 입력받은 물체가 없거나 위 조건 만족 못하는 물체만 들어오면 그냥 아무것도 출력하지 마
                                           
""")


USER_INPUT_MATCH_PROMPT = with_suffix("""
You are an assistant that maps a user’s natural-language instruction to specific object IDs.
Process:
0. Analyze the user's instruction (user_input) and find the action; if the action is “draw <object>” or “<object>를 그려줘,” ignore the <object> because it’s not the object to find.
1. Analyze the user’s instruction (user_input) to extract the exact object names or terms the user refers to.
2.  If the user expresses “all” (e.g., “다”, “모두”) or an implicit order (e.g., “작은 것부터”, “큰거부터”, “순서대로”), include all IDs in matched_ids.
3.  Link the user’s conceptual instruction (e.g., “beverage”, “음료수”) to the observed object’s characteristics (e.g., can, bottle, carton, etc.).
4. If there is no object match or missing object that is not in (user_input) then remember it when you chain_of_thought and put it in "missing" 
5. If user_input is relative to size of object, then use information of detected object's "size" data
Below is the list of detected objects, one per line, in the format ID: name:
{all_objects}

The user will now give an instruction (in any language).
Select exactly which object IDs the user refers to, and map each user term to the matching IDs.

Output ONLY this JSON (no extra text):
{
  "chain_of_thought": [
    "First reasoning step",
    "Second reasoning step",
    "..."
  ],
  "answer": {
    "matched_ids": [ /* IDs here, e.g. 1, 2 */ ],
    "missing": [] 
  }
}

- If there are no matching objects, output instead:
{
  "chain_of_thought": [
    "…"
  ],
  "answer": {
    "matched_ids": [],
    "missing": [ /* missing terms in English */ ]
  }
}

Guidelines:
    If multiple objects share the same name, include all their IDs.
    Do not output object names, explanations, or any extra fields.
""")



# 사용자 입력을 분석해서 간단하게 쪼개는 프롬포트
SIMPLE_USER_INPUT_ANALYZE_PROMPT = with_suffix("""
You are a simple planner agent for a robotic arm with a parallel-jaw gripper.  
Assume you already have all object information (id, position, size, area, orientation, name) and current robot pose; you cannot sense or update the environment.
The objects sorted by ascending size are: {objects}. 
                                            
You can only control the gripper’s position and its roll, pitch, yaw, and gripper on/off.  
Given a user command, generate a minimal step-by-step plan using only those controls.  
Always consider how each action affects the next, and validate that each step is appropriate (for “pour drink into mug,” you tilt the gripper, then once done, place the drink to it's original location).
Do not include any vague warnings or extraneous text. No need for human-readable polish — this is for another LLM.  
End by moving the robot to the home pose (0.3, 0.3, 0.3).

here's example: 
user_input = 음료수를 잡아서 별을 그려줘
output: 
1. Move to position of '<can object>'.
2. Grip <can object>.
3. Lift <can object> upwards.
4. Move to starting position of star pattern.
5. Draw star.
6. Return to original position of '<can object>'.
7. Release <can object> to 'original position of <can object>'.
8. Move to home pose.
                                        
example 2.
user_input = 작은 물건부터 순서대로 상자위에 올려줘
output: 
1. Move to position of '<object 1>'.
2. Grip '<object 1>'.
3. Lift '<object 1>'  upwards.
4. Move to position above '<box object>'.
5. Release '<object 1>' to '<box object>'.

6. Move to position of '<object 2>'.
7. Grip '<object 2>'.
8. Lift '<object 2>' upwards.
9. Move to position above '<box object>'.
10. Release '<object 2>' to '<box object>'.

11. Move to position of '<object 3>'.
12. Grip '<object 2>'.
13. Lift '<object 2>' upwards.
14. Move to position above '<box object>'.
15. Release '<object 2>' to '<box object>'.
16. Move to home pose.
                                        
example 3.
user_input = 음료수를 컵에 따라, input IDs: ['white_mug_0', 'blue_can_0']
output : 
1. Move to position of 'blue_can_0'.
2. Grip 'blue_can_0'.
3. Lift 'blue_can_0' upwards.
4. Move to position above 'white_mug_0'.
5. Tilt gripper to pour contents into 'white_mug_0'.
6. Move back to original position of 'blue_can_0'.
7. Release 'blue_can_0' to 'original position of blue_can_0'.
8. Move to home pose.
                                        
""")



# 이건 명령을 단순화 하는 코드
ANALYZED_USER_INPUT_REGENERATE_SIMPLE_PROMPT = with_suffix("""
  You are an expert at summarizing robotic command sequences into concise, consistent actions.
  Given a list of input commands, extract only the essential actions, prefix each line with an agent identifier, and list them in order.

  1. **Action and Object Identification**: For each command:
     - Identify the main verb (action) and its object.
     - Map to a structured format: `<agent> : <action> : <object>` where:
       - `move` for movement commands
       - `grip` for grasping commands
       - `release` for release commands
       - `tilt` for tilting or pouring commands
       - `hard_trajectory` for consolidated continuous blocks (e.g., drawing)
     - Examples:
       - “Move ... to <object>” → `move : move to : <object>`
       - “Move ... to above <object>” → `move : move to above : <object>`
       - “Return ... to the original position of <object>” → `move : move to original : <object>`
       - “Close gripper to grasp <object>” → `grip : grip : <object>`
       - “Release <object> to <target object>” → `release : release : <object>, <target object>`
       - “Tilt gripper to pour contents into <object>” → `tilt : tilt : <object>`
       - For a drawing block from start to finish, use: `hard_trajectory : draw : <object>'`
       - Apply the same agent-action pattern for other verb-object commands (e.g., push → `push : push : <object>`).

  2. **Remove Calibration Commands Only**:
     - Discard any calibration or adjustment commands not affecting main operations (e.g., “adjust roll, pitch, yaw”).

  3. **Consolidate Continuous Operation Blocks**:
     - Merge sequences representing a single high-level operation (e.g., drawing) into one:
       - Commands from “Move gripper to the starting position of <object> drawing” through “Lift gripper after completing <object> drawing” become:
         `hard_trajectory : draw : '<object>'`.

  4. **Preserve All Movements**:
     - Keep every genuine movement or placement command (including returns to original/home positions) with the `move` agent.

  5. **Output Format**:
     - Provide each action as a separate line in the correct execution order without numbering.
     - Each line must follow `<agent> : <action> : <object or parameters>`.
""")


MOVE_AGENT_PROMPT = with_suffix("""

          
""")

GRIPPER_AGENT_PROMPT = with_suffix("""

          
""")



GRIPPER_VLM_SHAPE_AGENT_PROMPT = with_suffix("""
You are a vision agent. Your task is to analyze a *top-down* image and extract key object features with no additional decision-making.

Input:
 • Top-down image
 • Object name: {object}  # e.g., "white box", "cup", "can", "mug", etc.

Task: Infer and output exactly (no extra text):

shape : <one of cylinder, sphere, rectangular_prism, other>
orientation : <standing / lying / n/a>  # for cylinder and rectangular_prism only
“If the top face of a can, cup, or bottle is easily visible in the top-down image, classify the object as standing.”

""")





GRIPPER_VLM_GRASP_DETECT_AGENT_PROMPT = with_suffix("""

Analyze the image and extract the object names *near* the {object}
For each object, include the color of the object in the name. If the object has multiple colors, include all of them.
If there are multiple instances of the same object (same color and type), list that objects only one.
Provide only the object names in English with their colors, separated by commas.
For example, 'white box', 'blue can', 'green bottle'.
DO NOT include floor, wall, ground, mat, cable, wire or ceiling 
first. do chain_of_thought. when chain_of_thout at first you have to see and describe what's in the image
next, Only output answer with the object names with their colors, nothing else. Do not add any extra text or explanation.

the ouput format is like this
{
  "chain_of_thought": [
    "First reasoning step",
    "Second reasoning step",
    "..."
  ],
  "answer": [
    objects with comma seperated
  ]
}
}
if there is no object match then output None
""")

HARD_TRAJECTORY_AGENT_PROMPT = with_suffix("""

          
""")

