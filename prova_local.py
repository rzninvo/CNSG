from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    dtype=torch.float16,
    device_map="auto"
)

# system_prompt = """
# You are a navigation assistant helping a user locate an object inside a building.
# Given a sequence of frames with detected objects, you must provide clear, human-like, easy-to-follow navigation instructions.

# Example:
# User: "Where is the sofa?"
# Observations:
# - In frame-000000 you see television (relative position: right, distance: close, room: living room)
# - In frame-000001 you see sofa (relative position: center, distance: close, room: living room)

# Assistant: "The sofa is in the living room near the television."

# End of example.
# """
system_prompt = """
You are a navigation assistant helping the user locate a target object inside a building.

You will receive a sequence of frames describing visible objects.  
Each object includes:  
- the floor,  
- the relative position to the viewer,  
- the distance from the viewer,  
- and the room it belongs to.

The frames appear in chronological order along the user's path from the starting point toward the target.

Before starting the walk description, consider an initial turn direction if provided.
Your task is to write a human-sounding description of the walk, fluent and easy to follow.  
Avoid technical language or numeric measurements. Use intuitive guidance and stay under 120 words (using fewer words when possible).

Mention at least one and at most two objects per room, choosing only the most informative for navigation.  
If the path includes stairs, simply write: “go up/down the stairs to reach the <room_name>”, without describing objects on the stairs.

If you see the target location or object, mention it immediately and stop referencing any further objects.

Only refer to objects that appear in the observations. Never invent or embellish details.  
When referencing an object, always include its ID (e.g., “chair_5”).

You will then receive a user question and the list of observations from the path, as well as the rooms visited in order. Imagine you are moving from the starting room to the target location, and provide clear path instructions.
"""

few_shots = """
### Example 1
User question: Where is the wall clock in the kitchen?
Observations:
In frame-000000, you see door_0 [(relative position: lower-right), (distance: close), (room: upper bedroom), (floor: 1)], couch_102 [(relative position: lower-left), (distance: mid-distance), (room: office), (floor: 1)].
In frame-000001, you see door_0 [(relative position: center-right), (distance: close), (room: upper bedroom), (floor: 1)], stairs_142 [(relative position: lower-center), (distance: close), (room: living room), (floor: 0)].
In frame-000002, you see stairs_142 [(relative position: lower-center), (distance: close), (room: living room), (floor: 0)], door_2 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)].
In frame-000003, you see door_2 [(relative position: center-left), (distance: close), (room: living room), (floor: 0)], armchair_59 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)].
In frame-000004, you see armchair_59 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)], couch_103 [(relative position: lower-left), (distance: close), (room: living room), (floor: 0)].
In frame-000005, you see wall clock_175 [(relative position: upper-left), (distance: slightly far), (room: kitchen), (floor: 0)], couch_103 [(relative position: lower-left), (distance: close), (room: living room), (floor: 0)].
Rooms visited in order: upper bedroom, office, living room, kitchen
Response:
Go down the stairs_142 in front of you, and reach the living room. Here you'll find an armchair_59 to your left and a couch_103 on your right. Continue straight ahead into the kitchen where the wall clock_175 is visible on the upper-left wall.
"""
extra_shots = """

### Example 2
User question: How do I get to the fireplace?
Observations:
In frame-000000, you see door_7 [(relative position: lower-left), (distance: close), (room: entryway), (floor: 0)], picture_114 [(relative position: center-left), (distance: close), (room: entryway), (floor: 0)].
In frame-000001, you see picture_114 [(relative position: center-left), (distance: very close), (room: entryway), (floor: 0)], flag_176 [(relative position: lower-right), (distance: close), (room: kitchen), (floor: 0)].
In frame-000002, you see chair_126 [(relative position: lower-left), (distance: close), (room: kitchen), (floor: 0)], door_6 [(relative position: center-right), (distance: close), (room: kitchen), (floor: 0)].
In frame-000003, you see fireplace_153 [(relative position: lower-left), (distance: slightly far), (room: living room), (floor: 0)], armchair_60 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)].
Rooms visited in order: entryway, kitchen, living room
Response:
From the entryway, reach the corridor where you see a flag_176 to enter the kitchen. Continue straight ahead into the living room, where the fireplace_153 is located to the lower-left.

### Example 3
User question: I want to go to the fireplace.
Observations:
Observations:
In frame-000000, you see fireplace_153 [(relative position: lower-left), (distance: slightly far), (room: living room), (floor: 0)], armchair_59 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
Rooms visited in order: living room
Response:
You can already see the fireplace_153, it's located right in front of you in the living room, just below the led tv_151.


### Example 4
User question: I am hungry, can you guide me to the refrigerator?
Observations:
In frame-000000, you see refrigerator_177 [(relative position: lower-left), (distance: mid-distance), (room: kitchen), (floor: 0)], kitchen cabinet_178 [(relative position: center-right), (distance: close), (room: kitchen), (floor: 0)].
Rooms visited in order: kitchen
Response:
The refrigerator_177 is located in front of you to your left.

### Example 5
User question: I want to go to the kitchen sink.
Observations:
In frame-000000, you see sink_184 [(relative position: lower-right), (distance: far), (room: kitchen), (floor: 0)], fireplace_153 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
Rooms visited in order: living room, kitchen
Response:
From the living room, head towards the kitchen, and you will find the sink_184 located to your lower-right.

### Example 6
User question: How do I reach the tv in the upper bedroom?
Observations:
In frame-000000, you see door_16 [(relative position: lower-center), (distance: close), (room: lower bedroom), (floor: 0)], door_14 [(relative position: lower-center), (distance: mid-distance), (room: entryway), (floor: 0)].
In frame-000001, you see door_14 [(relative position: lower-right), (distance: close), (room: entryway), (floor: 0)], door_15 [(relative position: center-left), (distance: close), (room: dining room), (floor: 0)].
In frame-000002, you see chair_152 [(relative position: lower-left), (distance: mid-distance), (room: kitchen), (floor: 0)], door_12 [(relative position: center-right), (distance: close), (room: kitchen), (floor: 0)].
In frame-000003, you see armchair_74 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)], couch_126 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
In frame-000004, you see armchair_73 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)], door_8 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)].
In frame-000005, you see door_8 [(relative position: lower-left), (distance: close), (room: living room), (floor: 0)], stairs_170 [(relative position: lower-right), (distance: close), (room: living room), (floor: 0)].
In frame-000006, you see picture_136 [(relative position: upper-right), (distance: close), (room: living room), (floor: 0)], picture_137 [(relative position: center-left), (distance: close), (room: living room), (floor: 0)].
In frame-000007, you see stairs_170 [(relative position: lower-center), (distance: mid-distance), (room: living room), (floor: 0)], couch_126 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
In frame-000008, you see stairs_170 [(relative position: center), (distance: very close), (room: living room), (floor: 0)], door_7 [(relative position: upper-left), (distance: slightly far), (room: office), (floor: 1)].
In frame-000009, you see door_3 [(relative position: upper-left), (distance: mid-distance), (room: upper bathroom), (floor: 1)], bicycle_119 [(relative position: center-right), (distance: close), (room: office), (floor: 1)].
In frame-000010, you see tv_30 [(relative position: lower-right), (distance: close), (room: upper bedroom), (floor: 1)], door_1 [(relative position: center-right), (distance: very close), (room: upper bedroom), (floor: 1)].
Rooms visited in order: lower bedroom, entryway, kitchen, living room, office, upper bedroom
Response:
Exit the lower bedroom through the door_16. Take the corridor on your right to the kitchen. Continue straight into the livingroom where you will find couch_126 on you right. From there, go up the stairs_170 to the first floor. Once upstairs, pass the bicycle and go through the door_1 to enter the upper bedroom, where the tv_30 is located to your lower-right.

### Example 7
User question: where is the sink in the kitchen?
Observations:
In frame-000000, you see sink_218 [(relative position: lower-left), (distance: mid-distance), (room: kitchen), (floor: 0)], armchair_74 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].
Rooms visited in order: kitchen
Response:
The sink_218 is at your lower-left in the kitchen, next to the kitchen cabinet_208.

### End of examples.
"""

# system_prompt += few_shots
# system_prompt += extra_shots

# user_prompt = """
# User question: where is the wall clock?

# Observations:
#     Initially, turn right.
# In frame-000000, you see wall clock_205 [(relative position: upper-center), (distance: mid-distance), (room: kitchen), (floor: 0)], couch_126 [(relative position: lower-left), (distance: close), (room: living room), (floor: 0)].

# Rooms visited in order: 
# living room, kitchen
# """

user_prompt = """
User question: where is the bicycle?

Observations:

In frame-000000, you see picture_140 [(relative position: center-left), (distance: close), (room: entryway), (floor: 0)], door_14 [(relative position: lower-right), (distance: mid-distance), (room: entryway), (floor: 0)].
In frame-000001, you see door_12 [(relative position: center-right), (distance: close), (room: kitchen), (floor: 0)], wall clock_205 [(relative position: upper-right), (distance: mid-distance), (room: kitchen), (floor: 0)].
In frame-000002, you see couch_126 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)], fireplace_182 [(relative position: lower-left), (distance: slightly far), (room: living room), (floor: 0)].
In frame-000003, you see door_8 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)], armchair_73 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)].
In frame-000004, you see door_8 [(relative position: lower-left), (distance: close), (room: living room), (floor: 0)], stairs_170 [(relative position: lower-right), (distance: close), (room: living room), (floor: 0)].
In frame-000005, you see picture_136 [(relative position: upper-right), (distance: close), (room: living room), (floor: 0)], picture_137 [(relative position: center-left), (distance: close), (room: living room), (floor: 0)].
In frame-000006, you see stairs_170 [(relative position: lower-center), (distance: mid-distance), (room: living room), (floor: 0)], picture_136 [(relative position: center-left), (distance: close), (room: living room), (floor: 0)].
In frame-000007, you see bicycle_119 [(relative position: lower-right), (distance: slightly far), (room: office), (floor: 1)], stairs_170 [(relative position: lower-center), (distance: close), (room: living room), (floor: 0)].

        Rooms visited in order: 
entryway, kitchen, living room, office 
"""

# user_prompt = """
# User prompt:
# User question: where is the fireplace?

# Observations:

# In frame-000000, you see fireplace_182 [(relative position: center-left), (distance: slightly far), (room: living room), (floor: 0)], armchair_73 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)].

# Rooms visited in order: 
# living room
# """

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]


# --- Create inputs ---
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

attention_mask = torch.ones_like(input_ids)

# --- Generate ---
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=500
    )

generated = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
