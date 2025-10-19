

BASE_PATH = "habitat-sim/data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/"
FILE_PATH = BASE_PATH+"TEEsavR23oF.semantic.txt"
OUTPUT_PATH = BASE_PATH+"semantic_info.txt"



def analyse_semantic_info(file_path, ignore_categories=None):
    semantic_info = {}
    with open(file_path, 'r') as f:
        for line in f:
            line_parts = line.strip().split(",")
            if len(line_parts) != 4:
                continue
            room_id = line_parts[3]
            category_id = line_parts[2].strip('"')

            if room_id not in semantic_info:
                semantic_info[room_id] = {}
            
            if category_id not in ignore_categories:
                if category_id not in semantic_info[room_id]:
                    semantic_info[room_id][category_id] = 1
                else:
                    semantic_info[room_id][category_id] += 1

            
    return semantic_info

if __name__ == "__main__":
    ignore_categories = ["ceiling", "floor", "wall", "handle", "window frame", "door frame", "frame", "unknown", ]
    semantic_info = analyse_semantic_info(FILE_PATH, ignore_categories)
    with open(OUTPUT_PATH, 'w') as output_file:
        for room_id, categories in semantic_info.items():
            print(f"Room ID: {room_id}", file=output_file)
            for category_id, count in categories.items():
                
                print(f"  Category ID: {category_id}, Count: {count}", file=output_file)
            

        different_categories = set(cat for room in semantic_info.values() for cat in room.keys())
        print("\n\nNumber of different categories (excluding ignored):", len(different_categories), file=output_file)
        for category in different_categories:
            print(" -", category, file=output_file)