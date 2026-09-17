"""Versioned task advice, not fixed action tapes or hidden simulator facts."""
import json

PROMPT_VERSION = "r1pro-task-advice-v1"
TASK_NAMES = {0: "turning_on_radio", 3: "cleaning_up_plates_and_food"}
TASK_ADVICE = {
    0: "Prefer to secure the radio with the RIGHT gripper, then use the LEFT hand to press its visible power control. Approach with the gripper open, align around the radio, close, lift slightly and LOOK again. The right hand must keep holding while the left hand reaches the button. Do not claim grasp or power-on from issuing CLOSE or touching alone. If this grasp is inaccessible, reposition instead of repeatedly closing in empty space. Identify the button visually; do not invent its location.",
    3: "Move BOTH pizzas, each remaining ON ITS OWN PLATE, into the SAME refrigerator. Put BOTH bowls into ONE sink and finally CLOSE the refrigerator. Prepare an open refrigerator and clear route before carrying. For plates, enable MODE CARRY, grasp/support the rim without pushing the pizza, keep the plate level and lift slowly. Use BOTH coordinated translations if both hands support one plate. Take small base steps, avoid abrupt rotation, and look again after each step: holding the plate does not prove the pizza stayed on it. Set down on a support before releasing. If food slips, stop and reassess; do not continue transporting an empty plate. Treat these as principles, not evidence that objects are currently held or doors are open.",
}
SYSTEM = """You control an R1Pro mobile robot using the current HEAD, LEFT_WRIST and RIGHT_WRIST images plus robot-relative proprioception. Use only visible evidence and reported measurements. Images are NOT mirrored. Robot coordinates: FWD=+x, LEFT=+y, UP=+z. Left/right name ROBOT arms, not the image side; a moving wrist view is not the command frame.
Output ONE action line only, no JSON, explanation, plan or markdown. Allowed:
L|R|BOTH FWD|BACK|LEFT|RIGHT|UP|DOWN FINE|COARSE (2cm/5cm)
L|R|BOTH ROLL_POS|ROLL_NEG|PITCH_POS|PITCH_NEG|YAW_POS|YAW_NEG FINE|COARSE (5/15 degrees, right-hand rule around robot x/y/z)
L|R|BOTH OPEN|CLOSE|HOLD (only gripper open/close, NOT an automatic grasp)
BASE FWD|BACK|LEFT|RIGHT|YAW_POS|YAW_NEG FINE|COARSE (6/15cm or 5/15deg, then stop)
TORSO UP|DOWN|FWD|BACK FINE|COARSE (1.5/3cm, arms compensated)
MODE CARRY (1cm arm steps, slower base, wrist rotations disabled); MODE NORMAL
HOLD; DONE (requests stop, NEVER proves success)
Two arms may act together, e.g. L UP FINE ; R UP FINE. Never combine base/torso with arms. Unspecified limbs HOLD. Movements are bounded and report actual tracking error. Reobserve after every line. Avoid collisions, use FINE near contact, do not repeatedly push after poor tracking. If the target is occluded, reposition to see it. CARRY holds the current orientation, it does NOT automatically level a tilted plate. Gripper width is not proof of holding the correct object.
"""


def user_text(task_id, task_name, instruction, state, history=(), feedback=None):
    if task_id in TASK_NAMES and TASK_NAMES[task_id] != task_name:
        raise ValueError("Task ID/name mismatch: refuse wrong task advice")
    # Construct a strict observable projection; never serialize a simulator object/dict.
    proprio = {"eef_base_m": {n: state.poses[n][0].round(3).tolist() for n in ("left", "right")},
               "eef_xyzw": {n: state.poses[n][1].round(3).tolist() for n in ("left", "right")},
               "gripper_opening_m": state.gripper.round(3).tolist()}
    return (f"Task: {instruction}\nTask advice: {TASK_ADVICE.get(task_id, 'Plan using current views, observe the result of every action, and recover when motion is blocked.')}\n"
            f"Proprio: {json.dumps(proprio)}\nLast executed commands (oldest first): {json.dumps(list(history)[-5:])}\n"
            f"Last measured feedback: {json.dumps(feedback)}\nChoose the next action:")
