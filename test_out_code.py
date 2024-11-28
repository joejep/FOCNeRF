import os

def gather_obj_feats(base_dir):
    obj_feats = {}
    target_dir = os.path.join(base_dir, "obj_feats")
    
    if os.path.isdir(target_dir):
        for file in os.listdir(target_dir):
            if file.endswith('.pt'):
                object_name = file.removesuffix('.pt')
                obj_feats[object_name] = os.path.join(target_dir, file)
                
    return obj_feats

# Usage
ckpt_dir = "/home/eiyike/FOCNeRF/results33/YOLO_TEST"
obj_feats = gather_obj_feats(ckpt_dir)
for obj_name, path in obj_feats.items():
    print(f'"{obj_name}" : "{path}"')