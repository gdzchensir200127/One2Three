import os
import glob
import re
from object_move_signal_simulate_only_signal import both_object_move_signal_simulate


def process_single_file(txt_file_path):
    print(f"[GPU-0] Processing file: {txt_file_path}")

    try:
        base_name = os.path.basename(txt_file_path)
        match = re.match(r"(\d+)_(.+)_(\d+)\.txt", base_name)
        if not match:
            print(f"[GPU-0] Skipping invalid file name: {base_name}")
            return

        idx, obj_name, inst_id = match.groups()

        # 构建路径
        scene_root = "E:/cjz/scene_ply_in"
        item_root = "E:/cjz/item_ply/real_obj_for_simulate"
        mask_root = "E:/cjz/mask_ply"
        mask_file = os.path.join(mask_root, "maskBoxMesh.ply")
        scene_file = os.path.join(scene_root, f"{idx}.ply")
        item_file = os.path.join(item_root, obj_name, inst_id, "models", "model_normalized.ply")

        print(f"Checking scene file: {scene_file}")
        print(f"Checking item file: {item_file}")

        if not os.path.exists(scene_file) or not os.path.exists(item_file) or not os.path.exists(mask_file):
            print(f"[GPU-0] Missing files for {base_name}, skipping...")
            return

        # 构建输出路径
        output_base = "E:/cjz/all_mask_output_signals_and_depths"
        out_dir = os.path.join(output_base, f"{idx}_{obj_name}_{inst_id}")

        both_object_move_signal_simulate(txt_file_path, scene_file, item_file, mask_file, out_dir)

        print(f"[GPU-0] Finished processing {base_name}")
    except Exception as e:
        print(f"[GPU-0] Error processing {txt_file_path}: {e}")
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈信息


if __name__ == '__main__':
    txt_dir = "E:/cjz/find_place_all"
    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))

    print(f"Found {len(txt_files)} .txt files to process.")

    # 串行处理文件
    for txt_file in txt_files:
        process_single_file(txt_file)

    print("All files processed.")
