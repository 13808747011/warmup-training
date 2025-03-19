import numpy as np
import nibabel as nib
import cc3d
import os
from scipy import ndimage

def remove_small_connections(img, min_size=10):
    """移除小的连通域"""
    processed_img = img.copy()
    # 对每个label单独处理
    for label in range(1, np.max(img) + 1):
        mask = (img == label)
        if not np.any(mask):
            continue
            
        # 进行连通域分析
        labeled = cc3d.connected_components(mask)
        stats = cc3d.statistics(labeled)
        
        # 移除小于min_size的连通域
        for region in range(1, np.max(labeled) + 1):
            if stats['voxel_counts'][region] < min_size:
                processed_img[labeled == region] = 0
                
    return processed_img

def detect_and_remove_weak_connections(processed_img, y_check, z_start, z_end, weak_threshold=3):
    """检测并移除非常小的弱连接，检查当前y的整个x层"""
    # 检查 y_check 处的整个 x 层区域
    weak_connection = processed_img[:, y_check:y_check + 1, z_start:z_end]
    if np.sum(weak_connection) <= weak_threshold:
        # 如果弱连接的体素数量小于等于阈值，则裁剪掉
        print(f"检测到弱连接，裁剪 z={z_start}-{z_end}, y={y_check}")
        processed_img[:, y_check:y_check + 1, z_start:z_end] = 0
        return True
    return False

def calculate_z_bounds(img, label_min=3, label_max=13):
    """计算z_min_global和z_max_global"""
    z_min_global = None
    z_max_global = None

    # 计算label=3的最小z值
    label_min_mask = (img == label_min)
    if np.any(label_min_mask):
        z_min_global = np.min(np.where(label_min_mask)[2])

    # 计算label=13的最大z值
    label_max_mask = (img == label_max)
    if np.any(label_max_mask):
        z_max_global = np.max(np.where(label_max_mask)[2])

    return z_min_global, z_max_global

def remove_weak_connections(img, z_window=1, y_window=5, y_step=6, y_min=100, y_max_limit=280):
    """通过分析局部z轴范围内的y向厚度来识别并移除弱连接"""
    processed_img = img.copy()
    z_size = img.shape[2]

    # 动态计算z_min_global和z_max_global
    z_min_global, z_max_global = calculate_z_bounds(img)
    if z_min_global is None or z_max_global is None:
        print("无法计算z_min_global或z_max_global，跳过处理")
        return processed_img

    for z in range(z_min_global, min(z_max_global + 1, z_size), z_window):
        z_start = max(z_min_global, z - z_window // 2)
        z_end = min(z_max_global + 1, z + z_window // 2 + 1)

        # 获取当前z窗口内的非零体素
        window_data = processed_img[:, y_min:y_max_limit, z_start:z_end]
        nonzero_coords = np.nonzero(window_data)
        
        if len(nonzero_coords[0]) > 0:
            # 找到最高点的坐标
            max_y_idx = np.argmax(nonzero_coords[1])
            x_target = nonzero_coords[0][max_y_idx]
            y_max = nonzero_coords[1][max_y_idx] + y_min

            current_y = y_max
            while current_y > y_min:
                y_check = current_y - y_step
                if y_check < y_min:
                    break

                # 检查从y_check到current_y范围内是否有体素
                check_slice = processed_img[x_target, y_check:y_check + 1, z_start:z_end]
                if np.any(check_slice):
                    # 找到体素，停止当前位置的裁剪
                    print(f"在z={z}, x={x_target}的y={y_check}处找到体素，停止裁剪")
                    break

                # 按y_window大小进行裁剪
                y_to_crop_end = current_y + 2  
                y_to_crop_start = max(y_check, y_min, current_y - y_window)  # 调整后的逻辑
                print(f"在z={z}裁剪y范围 {y_to_crop_start} - {y_to_crop_end}")
                processed_img[:, y_to_crop_start:y_to_crop_end, z_start:z_end] = 0
                current_y = y_to_crop_start

                # 检测并移除弱连接（在裁剪之后调用，检查整个x层）
                detect_and_remove_weak_connections(processed_img, y_to_crop_start, z_start, z_end)

    return processed_img

def main():
    input_dir = "AbdomenAtlasDemo"
    output_dir = "AbdomenAtlasDemo_cropped"
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
                    
                    # 应用新方法移除弱连接
                    processed_data = remove_weak_connections(img_data)
                    
                    case_output_dir = os.path.join(output_dir, case_dir)
                    if not os.path.exists(case_output_dir):
                        os.makedirs(case_output_dir)
                    
                    output_path = os.path.join(case_output_dir, filename)
                    processed_nib = nib.Nifti1Image(processed_data, img_nib.affine, img_nib.header)
                    nib.save(processed_nib, output_path)
                    
                    print(f"Saved cropped image to {output_path}")

if __name__ == "__main__":
    main()
