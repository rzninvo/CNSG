from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


d = {'bedroom_1': {'door': 2, 'chandelier': 1, 'wardrobe': 2, 'tv': 1, 'cabinet': 2, 'blanket': 1, 'pad': 1, 'bed': 1, 'pillow': 2, 'nightstand': 2, 'book': 2, 'table lamp': 2, 'plush toy': 2, 'window': 2, 'armchair': 1}, 'bathroom_1': {'mat': 1, 'towel': 1, 'bucket': 1, 'cabinet': 1, 'tap': 1, 'hand soap': 1, 'door': 2, 'toilet': 1, 'toilet brush': 1, 'lamp': 2, 'shower curtain': 1}, 'bedroom_2': {'curtain': 2, 'window': 2, 'photo': 1, 'sheet': 4, 'door': 2, 'toy': 10, 'board': 2, 'ventilation': 1, 'attic door': 1, 'chandelier': 2, 'light': 4, 'vent': 1, 'bicycle': 1, 'box': 6, 'couch': 2, 'basket': 2, 'wardrobe': 2, 'magazine': 2, 'book': 1, 'stack of papers': 10, 'picture': 1, 'folder': 1, 'table': 3, 'chair': 4, 'handbag': 2, 'pc tower': 1, 'trashcan': 1, 'computer desk': 1, 'plush toy': 2, 'printer': 1, 'telephone': 1, 'desk lamp': 1, 'plant': 3, 'shirt': 1, 'bag': 1, 'blanket': 1, 'newspaper': 2, 'balustrade': 1}, 'living_room': {'stairs': 32, 'picture': 7, 'door': 1, 'window curtain': 1, 'curtain rod': 3, 'speaker': 2, 'window': 2, 'curtain': 4, 'led tv': 1, 'fireplace': 1, 'flower': 4, 'chandelier': 1, 'armchair': 2, 'table': 3, 'decorative plate': 1, 'floor mat': 1, 'pillar': 1, 'fire alarm': 1, 'couch': 1, 'book': 1, 'lamp': 2, 'plate': 1, 'alarm control': 1, 'balustrade': 1}, 'kitchen': {'floor mat': 2, 'plate': 2, 'ceiling lamp': 3, 'light': 4, 'ceiling vent': 1, 'wall clock': 1, 'flag': 1, 'chair': 2, 'flower': 2, 'refrigerator': 1, 'kitchen cabinet': 3, 'curtain': 1, 'window': 1, 'kitchen appliance': 8, 'coffee machine': 2, 'coffee mug': 1, 'worktop': 1, 'sink': 1, 'tap': 1, 'knife holder': 1, 'microwave': 1, 'kitchen countertop item': 1, 'oven and stove': 1, 'fruit bowl': 1, 'dishwasher': 1}, 'office': {'door': 2}, 'bathroom_2': {'picture': 2, 'towel': 3, 'ventilation': 1, 'bath sink': 1, 'tap': 1, 'toilet paper': 2, 'toilet seat': 1, 'door': 1, 'door handle': 1, 'ceiling lamp': 1, 'hand soap': 1, 'bathroom shelf': 1}, 'laundry_room': {'bathroom shelf': 1, 'door': 3, 'alarm': 2, 'picture': 1, 'doormat': 1, 'window curtain': 2, 'window': 1, 'ventilation hood': 1, 'ceiling lamp': 2}, 'entryway': {'curtain': 2, 'window': 2, 'picture': 1, 'chair': 8, 'table': 1, 'cabinet': 1, 'flower': 3, 'pad': 2, 'ventilation hood': 1}, 'bedroom_3': {'chandelier': 1, 'curtain': 4, 'alarm': 1, 'curtain rod': 2, 'window': 2, 'dresser': 1, 'casket': 1, 'wall hanging decoration': 22, 'laundry basket': 2, 'led tv': 1, 'wardrobe': 1, 'electric box': 2, 'electrical controller': 1, 'bed': 2, 'pillow': 4, 'nightstand': 2, 'table lamp': 2, 'tissue box': 1, 'door': 1, 'ventilation hood': 1}, 'bathroom_3': {'ceiling vent': 1, 'ventilation hood': 1, 'bath mat': 3, 'trashcan': 1, 'door': 2, 'shower dial': 1, 'tap': 3, 'bath': 1, 'towel': 2, 'bathroom cabinet': 2, 'bathroom accessory': 3, 'mirror': 2, 'soap bottle': 1, 'mirror frame': 2, 'bath sink': 2, 'wall lamp': 4}, 'garage': {'shoe': 18, 'iron board': 1, 'iron': 1, 'clothes': 6, 'pillow': 3, 'blanket': 1, 'clothes hanger rod': 2, 'case': 1, 'stack of papers': 1, 'storage box': 1, 'briefcase': 2, 'backpack': 1, 'boxes': 1, 'door': 1, 'ceiling lamp': 1, 'ventilation hood': 1}, 'bathroom_4': {'door': 1, 'picture': 1, 'towel': 1, 'kitchen shelf': 1, 'bathroom shelf': 1, 'toilet paper': 1, 'bottle of soap': 1}}

messages = [
    {
        "role": "system",
        "content": (
            "You are an assistant for a home navigation system. "
            "You have a dictionary containing rooms and landmarks (objects) with the number of occurrences for each. "
            "The user may use synonyms or similar terms (e.g., 'clock' = 'wall clock'). "
            "Before responding, try to determine if there is a synonym or equivalent name in the dictionary. "
            "When the user asks to go to a room or mentions an object, follow these rules: "
            "1. If the requested room exists, return only the exact name of the room. "
            "2. If the object exists in only one room, return only the exact name of the object. "
            "3. If the object appears in multiple rooms, ask in which room it is located. "
            "4. If an object appears multiple times in the same room but not in others, return the name of the room. "
            "5. If no match is found, say you couldn't find the object and ask for more details. "
            "Always reply in the format: '<rule number>. <response text>'. "
            "Example: '1. kitchen' or '3. The object appears in multiple rooms, do you mean the one in kitchen or in dining room?'."
        )
    },
    {
        "role": "system",
        "content": f"Current dictionary of rooms and landmarks: {d}"
    },
    {"role": "user", "content": "Where is the clock?"}
]



# user_request = input("User: ")
# messages = [
#     {
#         "role": "system",
#         "content": (
#             "Extract only the main room or object that the user is referring to. "
#             "Return only one word or short phrase, exactly as it appears in the dictionary of rooms and landmarks. "
#             "Do not add explanations or punctuation. "
#             "Examples:\n"
#             "User: Where is the clock? → clock\n"
#             "User: Take me to the kitchen. → kitchen\n"
#             "User: Show me the tv in the bedroom. → tv\n"
#             "User: Go to the dining room. → dining room\n"
#             "User: Bring me to the bathroom. → bathroom"
#         )
#     },
#     {"role": "user", "content": user_request}
# ]



response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
)

print(response.choices[0].message.content)


# TODO to be tested for edge cases
# output = response.choices[0].message.content.strip().lower()
# goals, rooms, objects = [], [], []
# for key, value in d.items():
#     if output in key:
#         goal, room, object = key, key, None
#         goals = [goal]
#         rooms = [key]
#         objects = []
#     else:
#         for obj in value.keys():
#             if output in obj:
#                 goal, room, object = obj, key, obj
#                 goals.append(goal)
#                 rooms.append(key)
#                 objects.append(obj)


# print("Goals:", goals)
# print("Rooms:", rooms)
# print("Objects:", objects)
        