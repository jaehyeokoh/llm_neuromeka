"""
프롬포트 만들 때 주의할 점

1. 형식을 json같이 강요하면 성능이 매우 떨어진다 (json 형식의 문법 구조를 따르는데 리소스를 사용해 다른 품질이 떨어짐) 
1 해결법: 아래와 같이 파싱하기 쉽게 지정 후 나중에 출력을 파싱하면 된다 -> ### 이거로 구분, 파싱  코드는 utils.py의 parse_llm_non_json2json_output 사용하면 됨

- Avoid full natural language sentences unless needed for clarity.  
Output format:  
reasoning:  
<step-by-step logical reasoning- Include all justifications concisely>  
###  
answer:  
<comma-separated object names, or `None` if no match>





"""


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

그리고 "A물체를 제외한 나머지 물건들을 A위에 올려" 같은 명령은 단순 픽엔 플레이스니 simple agent에게 [A위에 A를 제외한 물건을 올린다]라는 블록 하나만 주면 되는거야.
그리고 만약 여러가지 동작이 있다면 나눠야해 예를 들어 "A와 B를 C 위에 올려줘" 라고 한다면 A를 C에 올려야 하고 B를 C에 올리는 총 2개의 액션으로 나누는거지 그러면 단순 옮기는 거니 simple agent에게 A를 C 위에 올려, B를 C위에 올려 이런식으로 나눌 수 있는거야
하지만 주의해 A를 잡아서 B에 올려는 [A를 잡아서 B에 올려라]지 [A를 잡는다], [A를 B에 올린다] 이런게 아니야 즉 잡아서 다른 위치에 놔두는거 까지가 하나의 액션인거야.
출력의 형식은 다음과 같아.

{
  "reasoning": [
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

DIVIDED_INPUT_EDITOR_PROMPT = """
WHO YOU ARE:
You are a divided_input editor agent. Your job is to refine vague action steps into concrete, executable instructions using the list of available objects. The agent information is managed separately — your focus is solely on rewriting each action using real object names.

INPUT:
- action_list: {action_list}
- objects on table: {objects}

YOUR GOAL:
Your goal is to rewrite each action so that:
- All vague references (like "an object", "another item") are replaced with concrete items from the table.
- If multiple objects can match, choose one based on reasonable inference (e.g., size, position, shape, or plausibility).
- The output answer are executable by a robot or physical agent with no ambiguity.
- Maintain the same number and order of answer in your output.

OUTPUT FORMAT:
reasoning:
Step 1: ...
Step 2: ...
###
answer:
steps:
 1. <Rewritten action> -- <comma separated related objects>
 2. <Rewritten action> -- <comma separated related objects>
  
RULES:
1. Do not generate or guess any new objects. Only use names exactly as listed in the "objects on table".
2. Maintain the order of the actions — output must match input numbering.
3. Each action should be one atomic instruction — do not combine multiple steps.
4. If an object reference is ambiguous or can't be resolved, explain in reasoning and skip that action.
5. Avoid full-sentence natural language unless necessary to explain ambiguities. Prefer concise commands.
6. If the instruction depends on object properties (e.g., size, weight, position) that are not provided in the object list, do not guess. Instead, return the original input action unchanged in the answer, and explain why in the reasoning.

EXAMPLE:
(example 1)
action_list:
1. Place one object to the right of the box.
2. Place another object on top of the box.
objects on table: white box, silver can, black cube
reasoning:
Step 1: 'box' is interpreted as 'white box' — only box-shaped object.
Step 2: Two objects needed; only 'silver can' and 'black cube' are available.
Step 3: Assign silver can to right-of-box action and black cube to top-of-box.
###
answer:
steps:
1. Place silver can to the right of the white box -- silver can, white box
2. Place black cube on top of the white box -- black cube, white box

(example 2)
action_list:
1. Place all objects in it's size
objects on table: white box, silver can, black cube
reasoning:
Step 1: the instruction depends on object properties. so returning original input
###
answer:
steps:
1. Place all objects in it's size --

"""



VLM_AGENT_PROMPT = with_suffix("""
Analyze the image and extract the object names
For each object, include the color of the object in the name. If the object has multiple colors, include all of them.
If any object names are in Korean, translate them into English.
If there are multiple instances of the same object (same color and type), list that objects only one.
Provide only the object names in English with their colors, separated by commas.
For example, 'white box', 'blue can', 'green bottle'.
DO NOT include floor, wall, groun, cable or ceiling 
Only output the object names with their colors, nothing else. Do not add any extra text or explanation.
""")

VLM_FEEDBACK_FAILED_OBJECTS_PROMPT = """
You are given a list of object names that failed to be detected by the vision system (GroundedSAM2), and the user's original instruction.

Your task is to review the failed object names and decide:
1. Which ones are still relevant to the user's intent.
2. Which ones need correction (e.g., misheard or invalid names or not good for GroundedSAM2 input (ex. blue-silver can -> blue can / silver can)).
3. Which ones should be discarded.


Rules:
- Only keep names that are clearly related to the user's instruction
- the output name must be in English.
- If a failed name is wrong (e.g., "fan" when the user meant "pen") or non_english(like korean), correct it to the right, detectable object name.
- Exclude irrelevant or invalid objects (e.g., "floor", "ceiling").
- Do not include any object name unless it comes from the failed list or is a corrected version of it.
- Do not invent new objects.
- Always include the color in the name, if available.
- Avoid full natural language sentences unless needed for clarity.
- All object names in the output must be translated into English, even if the original failed name was in another language such as Korean.
- For example, "흰 박스" → "white box", "파란 캔" → "blue can"

**Special Rule for Ambiguous Instructions:**
- If an object (e.g., "white box") is mentioned in a way that excludes it from manipulation ("put everything except A"), but it also appears as a destination or location ("on A", "next to A"), then: **Do NOT discard it. Include it in the output, because it is still needed for the task (as a reference object).**
  → Always include such objects if they are required to **complete the action** (e.g., placing something *on* it).

If no object should be re-detected, return an empty string.

Here are the failed object names:
{detect_failed_objects}

Here is the user's input:
{user_input}

Output format:  
reasoning:  
<step-by-step logical reasoning- Include all justifications concisely>  
###  
answer:  
<comma-separated object names, or `None` if no match>


"""




# 사용자 입력을 분석해서 간단하게 쪼개는 프롬포트
SIMPLE_USER_INPUT_ANALYZE_PROMPT = """
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
3. Move to starting position of star pattern.
4. Draw star.
5. Return to original position of '<can object>'.
6. Release <can object> to 'original position of <can object>'.
7. Move to home pose.
                                        
example 2.
user_input = 작은 물건부터 순서대로 상자위에 올려줘
output: 
1. Move to position of '<object 1>'.
2. Grip '<object 1>'.
3. Move to position above '<box object>'.
4. Release '<object 1>' to '<box object>'.

5. Move to position of '<object 2>'.
6. Grip '<object 2>'.
7. Move to position above '<box object>'.
8. Release '<object 2>' to '<box object>'.

9. Move to position of '<object 3>'.
10. Grip '<object 2>'.
11. Move to position above '<box object>'.
12. Release '<object 2>' to '<box object>'.
13. Move to home pose.
                                        
example 3.
user_input = 음료수를 컵에 따라, input IDs: ['white_mug_0', 'blue_can_0']
output : 
1. Move to position of 'blue_can_0'.
2. Grip 'blue_can_0'.
3. Move to position above 'white_mug_0'.
4. Tilt gripper to pour contents into 'white_mug_0'.
5. Move back to original position of 'blue_can_0'.
6. Release 'blue_can_0' to 'original position of blue_can_0'.
7. Move to home pose.

Make sure to think step by step when answering 
now Begin.
"""

# SIMPLE_USER_INPUT_ANALYZE_PROMPT의 출력을 검토하는 프롬포트
SIMPLE_PLAN_EDITOR_PROMPT = """
You are a simple plan editor agent that validates and adjusts a robotic arm plan based on the user's natural-language instruction.

Your input:
1. a plan generated from the user input = {plan}
2. the user's raw instruction in natural language = {user_input}

Your Goals:
1. Check the plan for object name validity. If any object name is not in the correct ID format (e.g., <box object> or blue can), do not fix it. Instead, return it in missing.
2. If the instruction includes spatial direction terms (e.g., "옆에", "왼쪽", "behind"), normalize them and update the plan accordingly.

-----------------------------------------
Object Name Rules:
- Valid format: e.g., 'blue_can_0', 'white_box_1' (must have suffix like `_0`)
- Invalid: e.g., 'blue can', 'white box', or placeholders like `<box object>` or '<can object>'
- You must NOT guess or fix object names — only detect them and return in `missing`.

Direction Normalization Rules:
- Normalize all spatial direction phrases to one of the following ONLY:
  - "on" (위에, on, on top)
  - "next to" (옆에, beside)
  - "left" (왼쪽, left of)
  - "right" (오른쪽, right of)
  - "front" (앞에, in front of)
  - "behind" (뒤에, back of)

Use these keywords **only** in the modified plan.

-----------------------------------------
Output Rules:

1. If a wrong or placeholder object name is used in the plan → return missing list  
2. If all object names are valid → modify plan as needed (e.g., direction)

-----------------------------------------
Output format:

If any object is invalid:

reasoning:
<step-by-step justification>
###
answer:
missing: ["object_name_1", "object_name_2"]

If all objects are valid and the plan only needs spatial fix:

reasoning:
<step-by-step justification>
###
answer:
1. Step one...
2. Step two...

-----------------------------------------
Example 1:
User input: "음료수를 박스 옆에 놔줘"
Plan:
1. Move to position of 'blue_can_0'.
2. Grip 'blue_can_0'.
3. Move to position above '<box object>'.
4. Release 'blue_can_0' to '<box object>'.

→ Output:

reasoning:
1. The object '<box object>' is not a valid ID.
2. The user said "옆에" which maps to "next to".
3. Cannot continue until correct object ID is given.
###
answer:
missing: ["box object"]

-----------------------------------------
Example 2:
User input: "음료수를 박스 옆에 놔줘"
Plan:
1. Move to position of 'blue_can_0'.
2. Grip 'blue_can_0'.
3. Move to position above 'white_box_0'.
4. Release 'blue_can_0' to 'white_box_0'.

→ Output:

reasoning:
1. All object names are valid.
2. The instruction includes "옆에", which maps to "next to".
3. Step 4 uses "above" → must be updated to "next to".
###
answer:
1. Move to position of 'blue_can_0'.
2. Grip 'blue_can_0'.
3. Move to position next to 'white_box_0'.
4. Release 'blue_can_0' to 'white_box_0'.
5. Move to home pose.

"""


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
      - "Move to position of <object>" → `move : to : <object>`
      - "Move to position to the right of <object>" → `move : right : <object>`
      - "Move to position to the left of <object>" → `move : left : <object>`
      - "Move to position above <object>" → `move : above : <object>`
      - "Move to position front of <object>" → `move : front : <object>`
      - "Move to position back of <object>" → `move : behind : <object>`                                                   
      - "Return to the original position of <object>" → `move : original : <object>`
      - "Move to home pose" → `move : home : none`
      - "Close gripper to grasp <object>" → `grip : grasp : <object>`
      - "Release <object> to <target object>" → `release : to : <object>, <target object>`
      - "Release <object> to the right of <target>" → `release : right : <object>, <target>`
      - "Tilt gripper to pour contents into <object>" → `tilt : pour : <object>`
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



GRIPPER_VLM_GRASP_DETECT_AGENT_PROMPT = """
Analyze the image. You are given a list of target objects: {object}.

Step-by-step:
1. Briefly describe what objects are visually present in the image.
2. For each detected object, compare it with the given target list.
3. Reason step by step using short but complete logical expressions:
   - Justify every match or exclusion.
   - Do not skip any reasoning step, even if it seems obvious.
   - Avoid full natural language sentences unless clarity requires it.

4. When comparing object names, consider both **object type** and **visual features**:
  - Match objects if the **object type** (e.g., "can", "box") clearly matches.
  - Allow **partial color matches**, but do not require exact color.
  - If the detected object is clearly the same type (e.g., a can), and the **color is reasonably similar or related**, consider it a match.
  - Prioritize **object type and visual similarity**, not exact text match.


5. If any visually present object is **not** in the target list, include it too using the following rules:
   - Describe each with a single dominant color (e.g., "white box", "blue can").
   - List each object only once, even if multiple identical ones are seen.
   - **Do not include** non-graspable or irrelevant items such as: mat, floor, wall, wire, cable, ceiling.

---

Output format:  
reasoning:  
<short reasoning steps>  
###  
answer:  
<object1, object2, ...>

- If no relevant objects are found, output:  
  answer:  
  None

"""
########################################################## 여기서부터 Hard Agent #################################################
# 어려운 task받아 적절한 에이전트에게 분배하는 프롬포트
HARD_AGENT_SUPERVISOR_ASSIGN_PROMPT = """
WHO YOU ARE:
You are a supervisor agent responsible for assigning user_input to the correct specialized agent.

AGENTS:
- Assembly_agent: Handles requests to build or assemble structures or object shapes.
- Analysis_agent: Handles vague, unclear, or ambiguous user instructions.
- Hard_Trajectory_agent: Handles tasks involving complex or irregular movement paths.
- Reasoning_agent: General-purpose reasoning agent used when no other category applies.

TASK:
Analyze the following user_input and assign it to the most appropriate agent from the list above.

INPUT:
user_input: {plan_input}

INSTRUCTIONS:
1. Do not modify or rephrase the input.
2. Classify the input into one of the four categories based on intent.
3. Match it to the corresponding agent.
4. Use concise reasoning — bullet-style or minimal phrases.
5. Output must follow the format below.

OUTPUT FORMAT:
reasoning:
Step 1: [What keyword or phrase is relevant]
Step 2: [Which category it belongs to and why]
###
answer:
[Agent_Name]

EXAMPLES:

Example 1:
user_input: "앞에 있는 거로 고인돌 모양 만들어줘"

reasoning:
Step 1: "모양 만들어줘" implies object assembly  
Step 2: Classified as Assembly → use Assembly_agent
###
answer:
Assembly_agent

Example 2:
user_input: "상자를 잡아서 지그재그로 움직여줘"

reasoning:
Step 1: "지그재그로 움직여줘" implies complex motion  
Step 2: Classified as Hard_Trajectory → use Hard_Trajectory_agent
###
answer:
Hard_Trajectory_agent

RULE:
Return only the reasoning and final agent name. Do not add extra text or explanations.
"""