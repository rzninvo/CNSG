from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    dtype=torch.float16,
    device_map="auto"
)

system_prompt = """
You are a navigation assistant helping a user locate an object inside a building.
Given a sequence of frames with detected objects, you must provide clear, human-like, easy-to-follow navigation instructions.

Example:
User: "Where is the sofa?"
Observations:
- In frame-000000 you see television (relative position: right, distance: close, room: living room)
- In frame-000001 you see sofa (relative position: center, distance: close, room: living room)

Assistant: "The sofa is in the living room near the television."

End of example.
"""


user_prompt = """
Where is the wall clock?

Observations from the path:
       
In frame-000000, you see door_0 [(relative position: center-right), (distance: close), (room: upper bedroom), (floor: 1)], stairs_142 [(relative position: lower-center), (distance: close), (room: living room), (floor: 0)].
In frame-000001, you see stairs_142 [(relative position: lower-center), (distance: close), (room: living room), (floor: 0)], door_2 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)].
In frame-000002, you see door_2 [(relative position: center-left), (distance: close), (room: living room), (floor: 0)], armchair_59 [(relative position: lower-left), (distance: mid-distance), (room: living room), (floor: 0)].
In frame-000003, you see armchair_59 [(relative position: lower-right), (distance: mid-distance), (room: living room), (floor: 0)], couch_103 [(relative position: lower-left), (distance: close), (room: living room), (floor: 0)].
In frame-000004, you see wall clock_175 [(relative position: upper-left), (distance: slightly far), (room: kitchen), (floor: 0)], couch_103 [(relative position: lower-left), (distance: close), (room: living room), (floor: 0)].
In frame-000005, you see wall clock_175 [(relative position: upper-left), (distance: slightly far), (room: kitchen), (floor: 0)], couch_103 [(relative position: lower-left), (distance: close), (room: living room), (floor: 0)].
In frame-000006, you see wall clock_175 [(relative position: upper-center), (distance: mid-distance), (room: kitchen), (floor: 0)], chair_126 [(relative position: lower-right), (distance: mid-distance), (room: kitchen), (floor: 0)].
"""

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
        max_new_tokens=50
    )

generated = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
