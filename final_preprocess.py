import numpy as np
import nibabel as nib
import cc3d
import os

def process_small_vertebrae(img):
    """处理小尺寸椎骨"""
    mod_img = img.copy()
    
    # 逐个检查label 1-5
    for label in range(1, 6):
        total_voxels = np.sum(img == label)
        
        if total_voxels < 50000:
            print(f"Label {label} has {total_voxels} voxels, less than 50000")
            # 将当前label及后4个label的最大连通域依次改为label-1到label+3
            for i, target_label in enumerate(range(label, label + 5)):
                mask = mod_img == target_label
                if np.any(mask):  # 如果存在这个label
                    cc_labels = cc3d.connected_components(mask)
                    if np.max(cc_labels) > 0:  # 确保有连通域
                        sizes = [(i, np.sum(cc_labels == i)) for i in range(1, np.max(cc_labels) + 1)]
                        if sizes:  # 确保有连通域
                            largest_cc_id = max(sizes, key=lambda x: x[1])[0]
                            new_label = target_label - 1  # 每个标签都变为它前一个标签
                            mod_img[cc_labels == largest_cc_id] = new_label
                            print(f"Changed label {target_label}'s largest component to {new_label}")
            
            # 一旦找到一个小于50000的label，就停止检查
            break
    
    return mod_img

def main():
    input_dir = "AbdomenAtlasDemo_cropped"
    output_dir = "AbdomenAtlasDemo_preprocessed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for case_dir in os.listdir(input_dir):
        case_path = os.path.join(input_dir, case_dir)
        if os.path.isdir(case_path):
            for filename in os.listdir(case_path):
                if filename.endswith(".nii.gz"):
                    print(f"Processing {filename} in {case_dir}")
                    
                    filepath = os.path.join(case_path, filename)
                    img_nib = nib.load(filepath)
                    img_data = img_nib.get_fdata().astype(np.uint8)
                    
                    # 处理图像
                    processed_data = process_small_vertebrae(img_data)
                    
                    case_output_dir = os.path.join(output_dir, case_dir)
                    if not os.path.exists(case_output_dir):
                        os.makedirs(case_output_dir)
                    
                    output_path = os.path.join(case_output_dir, filename)
                    processed_nib = nib.Nifti1Image(processed_data, img_nib.affine, img_nib.header)
                    nib.save(processed_nib, output_path)
                    
                    print(f"Saved preprocessed image to {output_path}")

if __name__ == "__main__":
    main()
