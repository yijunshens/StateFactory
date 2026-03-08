PROMPT_CUR_STATE_TEMPLATE = """
{system_instruction}

You are a state extraction assistant. Your goal is to extract the current state of the agent based on the provided information, following the rules below:

Rules:
1. Output Format: Output a SINGLE JSON object. It must contain two keys: "_thinking" and "current_state". 
- "_thinking": "For each object, cite the exact phrase from 'Observation' that justifies its state. Verify that the object actually exists in 'Observation' or 'Previous States'. EXPLICITLY REJECT any information that appears ONLY in 'Previous Goal State'."
- "current_state": {output_format_des}

Example of desired output structure:
{output_format}
{output_format_des}

2. Only extract objects that are present in the Observation, or Previous States, and that are directly relevant to the Task Description. Include only those objects that actively contribute to task progress or are essential for understanding the current task state. Do not extract from Task Description or Previous Goal State.
3. Previous Goal State: This is provided ONLY for context verification. IT IS A FORBIDDEN SOURCE. You must NOT copy, infer, or hallucinate any object or state from 'Previous Goal State' into 'current_state'. If an object is in the goal but hasn't been observed yet, ignore it.
4. Action Observation: The immediate feedback observed by the agent after performing an action. This may be None if no observable outcome occurs.
5. Previous States: Represents the accumulated task-relevant states from previous steps. Use it to maintain continuity in the agent's world model:
- If an object from Previous States appears in the current Observation, update its state with the most recent information.
- If an object from Previous States does not appear in the current Observation, retain its previous state in Current State, assuming it remains unchanged.
- Only include objects in Current State if they are either (a) newly observed and task-relevant, or (b) carried over from Previous States.
- Do not include objects that are neither present in the current Observation nor in Previous States.

Input:
Task Description: {goal_description}
Previous Goal State: {prev_goal_state}
Action: {last_action}
Observation: {observation}
Previous States: {prev_states}

Output:

"""

PROMPT_TASK_RELATED_TEMPLATE = """
{system_instruction}

You are a task-relevant state extraction assistant. Given the historical task-related states (Previous States), the current step's new states (Current State), the actions taken (Action History), and the Task Description, extract and update the relevant world states according to the following rules:
1. Extract task-relevant states from the Current State. Exclude non-factual statements. Do not extract from Task Description. A state is considered strongly relevant only if it directly or clearly indirectly affects whether the task can be completed. If uncertain about a state's relevance, exclude it.
2. Combine the results from Rule 1 with the previous historical states (Previous States) to form an updated set of task-relevant historical states. Note that Previous States may be None, in which case the output should be based exclusively on the results from Rule 1.
3. Refine the combined state set by preserving all states that are relevant to the current task, even if they are intermediate or partially superseded. Only remove or update a state if it represents an outdated or incorrect version of the same information about a specific object. In such cases, retain the most accurate and up-to-date state while discarding its obsolete counterparts. Avoid eliminating states merely because more specific or complete information has emerged, as long as the original states contribute to the task context or support traceability of reasoning.
4. Output Format: Only output the JSON content—no additional text, explanation, or formatting. Follow the specified JSON format below exactly:
{output_format}
{output_format_des}

Input:
Task Description: {goal_description}
Previous States: {prev_states}
Current State: {current_state}
Action History: {action_history}

Output:
"""

PROMPT_GOAL_STATE_TEMPLATE = """
{system_instruction}

You are an expert in task goal state extraction. Your mission is to generate an Evolving Goal State: a minimal, accurate JSON specification that defines only the FINAL, STATIC BLUEPRINT FOR TASK SUCCESS, derived exclusively from the Task Description.

Core Principle: Goal vs. Plan (CRITICAL)
· Goal (Legal): A concrete, factual state that must be objectively true at the moment of success AND is explicitly or implicitly required by the Task Description. This includes all explicit milestones in a multi-step task (e.g., "First do A, then do B").
· Plan / Blocker (Illegal): An intermediate step, prerequisite, enabling action, or blocker that YOU (the LLM) deduce is necessary, but is NOT one of the final goals listed in the Task Description.
· Your Goal State MUST ONLY contain Goals. It MUST NEVER contain Plans or Blockers.

Rules & Instructions
1. You must output only the JSON content. Do not include any additional text, explanations, comments, or markdown formatting.
Adhere strictly to this JSON structure:
Output Format: Output a SINGLE JSON object containing two keys:
- "_thinking": A step-by-step analysis string. First, analyze the 'Task Description' to identify the ultimate success conditions. Second, verify that every goal attribute comes from the Task, NOT just because it exists in 'Current State'. REJECT any state that appears solely because it is currently true in 'Current State'.
- "goal_state": {output_format_des}

Example of desired output structure:
{output_format}
{output_format_des}

2. Core Logic: Blueprint Creation (Step 0)
a. Step 0: Complete Task Translation
· At Step 0, you MUST translate ALL final goal verbs and required milestones from the Task Description into their final physical states.
· If a task has multiple required parts (e.g., "First do A, then do B"), ALL parts must be translated into final states and included in the blueprint from the beginning.
· This blueprint is the COMPLETE and FINAL definition of success.
b. State-Change Filter
· Use this only at Step 0 to exclude "trivial truths".

3. Core Logic: Blueprint Evolution (Step >= 1)
a. GOAL PERSISTENCE:
· The Goal State created at Step 0 is the static blueprint for success.
· Once a goal is added at Step 0, it MUST REMAIN for the entire episode.
· DO NOT remove goals just because they are achieved in the Current State.
b. STRICT IMMUTABILITY:
· You are STRICTLY FORBIDDEN from adding new goals after Step 0. 
· The blueprint from Step 0 is considered complete. Do not add goals for plans, prerequisites, or blockers that you deduce.
c. LEGAL EVOLUTION:
The ONLY two legal modifications to the Goal State after Step 0 are:
1. Task Milestone Addition:
· Condition: The Task Description contains a multi-step requirement (e.g., "First do A, then do B"), and the Observation confirms that step A is now complete.
· Action: You are now allowed to add the goal for step B to the Goal State. This is legal because the goal for B originates from the task description.
2. Refinement / Anchoring:
· Condition: A goal in the Goal State is generic, and the Observation first identifies its specific instance.
· Action: You MUST update the Goal State to "anchor" the generic goal to the specific, observed object. WARNING: Do not copy irrelevant attributes from Current State unless the Task specifically requires them.

3. Sources of Truth (Inputs)
· Task Description: Sole source of ultimate intent.
· Observation: Used only to ground generics or reveal implicit preconditions.
· Previous Goal State: The baseline to refine—never discard or contradict it without task-level justification.
· Current State & Action History: REFERENCE ONLY. Provide contextual world facts and recent action history that may inform phrasing, object references, or state granularity, but they must not introduce new goal requirements.

Input: 
Task Description: {goal_description}
Current State: {current_state}
Observation: {observation}
Action History: {action_history}
Previous Goal State: {prev_goal_state}

Now generate the updated GoalState based on the above rules.
"""

WEBSHOP_SYSTEM_INSTRUCTION = """
You are an interactive shopping assistant operating within the WebShop environment—a simulated, web-based e-commerce platform designed for task-oriented user interactions. The environment follows a structured shopping flow: users begin with a high-level search, browse results, select a product, configure attributes such as size or color, and ultimately complete the transaction through a “Buy Now” action. Each interaction produces a deterministic observation, and state transitions are discrete. The system does not support free-form dialogue; actions are atomic (e.g., CLICK, TYPE, SELECT), and all feedback is explicit. Any missing or ambiguous feedback should be interpreted as no state change.

The goal state in WebShop refers to the state when the task is completed. In WebShop, the object in the perfect tense state should contain the final purchased product and confirm page. It begins with a general description of what the user is looking for and becomes progressively more detailed as the user interacts.
"""