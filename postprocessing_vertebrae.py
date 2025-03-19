import numpy as np
import nibabel as nib
import cc3d
import copy
import os
import logging
import argparse
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default='/root/autodl-tmp/warmup_training/data/after_specialprocessing/', help="Directory containing .nii.gz files to be postprocessed")
parser.add_argument("--target_dir", type=str, default='/root/autodl-tmp/warmup_training/data/final_processing/', help="Directory containing .nii.gz files to be postprocessed")
parser.add_argument("--nof_jobs", type=int, default=1, help="Nof jobs for parallel processing the files")

# 使用新的标签定义
vertebrae_labels = {
    "vertebrae C1": 24,
    "vertebrae C2": 23,
    "vertebrae C3": 22,
    "vertebrae C4": 21,
    "vertebrae C5": 20,
    "vertebrae C6": 19,
    "vertebrae C7": 18,
    "vertebrae T1": 17,
    "vertebrae T2": 16,
    "vertebrae T3": 15,
    "vertebrae T4": 14,
    "vertebrae T5": 13,
    "vertebrae T6": 12,
    "vertebrae T7": 11,
    "vertebrae T8": 10,
    "vertebrae T9": 9,
    "vertebrae T10": 8,
    "vertebrae T11": 7,
    "vertebrae T12": 6,
    "vertebrae L1": 5,
    "vertebrae L2": 4,
    "vertebrae L3": 3,
    "vertebrae L4": 2,
    "vertebrae L5": 1
}

def get_index_arr(img):
    return np.moveaxis(np.moveaxis(np.stack(np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]))),0,3),0,1)

def get_relevant_ccs(cc, keep_threshold, keep_main=True):
    if keep_main:
        cutoff_idx = 1
    else:
        cutoff_idx = 2
    return sorted([(x,np.sum(cc==x)) for x in np.unique(cc) if np.sum(cc==x) > keep_threshold],key=lambda x:x[1], reverse=True)[cutoff_idx:]

def merge_cc_of_adjacent(cc_cur, cc_above, voxel_supression_threshold):
    """合并相邻椎骨的连通域"""
    nof_voxels_cc = [(x, np.sum(cc_cur == x)) for x in np.unique(cc_cur)]
    relevant_cc = []

    for idx, nof_voxels in nof_voxels_cc:
        if nof_voxels > voxel_supression_threshold:
            relevant_cc.append((idx, nof_voxels))
    
    # 假设背景是最大的连通域，从relevant_cc中移除
    relevant_cc = sorted(relevant_cc, key=lambda x: x[1], reverse=True)[1:]
    
    nof_voxels_above = [(x, np.sum(cc_above == x )) for x in np.unique(cc_above)]
    relevant_cc_above = []
    for idx, nof_voxels in nof_voxels_above:
        if nof_voxels > voxel_supression_threshold:
            relevant_cc_above.append((idx, nof_voxels))
    
    # 忽略最大的非背景连通域，因为它将是椎骨本身
    relevant_cc_above = sorted(relevant_cc_above, key=lambda x: x[1], reverse=True)[2:]

    if len(relevant_cc_above) > 0:
        # 将上部剩余组件与当前椎骨的所有相关cc合并
        mskcc_pool = np.zeros(cc_cur.shape).astype(np.bool_)
        for idx, _ in relevant_cc_above:
            mskcc_pool = np.logical_or(mskcc_pool, cc_above==idx)
        for idx, _ in relevant_cc:
            mskcc_pool = np.logical_or(mskcc_pool, cc_cur == idx)

        cc_pool = cc3d.connected_components(mskcc_pool, connectivity=6)  # 添加connectivity=6
        rel_components_pool = sorted([(x, np.sum(cc_pool == x )) for x in np.unique(cc_pool)],key=lambda x:x[1], reverse=True)[1:]

        return cc_pool==rel_components_pool[0][0]
    else:
        return None

def spine_adjacent_pairs(img, voxel_supression_threshold=10, default_val=0, include_sacrum=True):
    """检查相邻连通域以识别错误分配到的椎骨"""
    labels = sorted(list(vertebrae_labels.values()), reverse=True)
    if include_sacrum:
        labels.append(102)
    
    mod_img = copy.deepcopy(img)

    # 获取相邻椎骨的三元组
    triplets = []
    for l in range(len(labels)):
        # 常规三元组
        if l > 0 and l < len(labels)-1:
            triplets.append((labels[l-1], labels[l], labels[l+1]))
        # 第一个三元组
        elif l<len(labels)-1:
            assert l == 0, "确保是第一个元素"
            triplets.append((labels[l], labels[l+1]))
        # 最后一个三元组
        elif l>0:
            assert l==len(labels)-1, "确保是最后一个元素"
            triplets.append((labels[l-1], labels[l]))
    
    for idx, triplet in enumerate(triplets):
        if idx==0 or idx==len(triplets)-1:
            current, below = triplet
            above = None
        elif idx == len(triplets)-1:
            above, current = triplet
            below = None
        else:
            above, current, below = triplet
            
        msk_cur = mod_img == current
        cc_cur = cc3d.connected_components(msk_cur, connectivity=6)  # 添加connectivity=6
        
        # 抑制小连通域
        nof_voxels_cc = [(x, np.sum(cc_cur == x)) for x in np.unique(cc_cur)]
        relevant_cc = []

        for idx, nof_voxels in nof_voxels_cc:
            if nof_voxels > voxel_supression_threshold:
                relevant_cc.append((idx, nof_voxels))
            else:
                mod_img[cc_cur == idx] = default_val
        
        background_index = sorted(relevant_cc, key=lambda x: x[1], reverse=True)[0]
        relevant_cc.remove(background_index)

        if above is not None:
            msk_above = mod_img == above
            cc_above = cc3d.connected_components(msk_above, connectivity=6)
            rel_cc_above = get_relevant_ccs(cc_above, keep_threshold=voxel_supression_threshold, keep_main=False)
        
        if below is not None:
            msk_below = mod_img == below
            cc_below = cc3d.connected_components(msk_below, connectivity=6)
            rel_cc_below = get_relevant_ccs(cc_below, keep_threshold=voxel_supression_threshold, keep_main=False)
        
        if above is not None and len(rel_cc_above) > 0:
            consolidated_vetebra_above = merge_cc_of_adjacent(cc_cur, cc_above, voxel_supression_threshold=voxel_supression_threshold)
            if consolidated_vetebra_above is not None:
                mod_img[consolidated_vetebra_above] = current
        elif below is not None and len(rel_cc_below) > 0:
            consolidated_vetebra_below = merge_cc_of_adjacent(cc_cur, cc_below, voxel_supression_threshold=voxel_supression_threshold)
            if consolidated_vetebra_below is not None:
                mod_img[consolidated_vetebra_below] = current
    
    return mod_img

def supress_non_largest_components(img, default_val = 0):
    """专注于椎骨的连通域处理"""
    logger = get_this_logger()
    index_arr = get_index_arr(img)
    img_mod = copy.deepcopy(img)
    new_background = np.zeros(img.shape, dtype=np.bool_)
    
    for label in vertebrae_labels.values():
        label_cc = cc3d.connected_components(img == label, connectivity=6)
        uv, uc = np.unique(label_cc, return_counts=True)
        dominant_vals = uv[np.argsort(uc)[::-1][:2]]
        if len(dominant_vals)>=2:
            new_background = np.logical_or(new_background, 
                np.logical_not(np.logical_or(label_cc==dominant_vals[0], 
                label_cc==dominant_vals[1])))
                
    for voxel in index_arr[new_background]:
        img_mod[tuple(voxel)] = default_val
    return img_mod

def get_this_logger(print_to_console=True):
    logger = logging.getLogger("postprocessing")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("postprocessing.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    if print_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def get_error_logger():
    logger = logging.getLogger("error")
    logger.setLevel(logging.ERROR)
    fh = logging.FileHandler("error.log")
    fh.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levellevel)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def bodypart_limitations(img):
    """简化的位置处理 - 仅处理椎骨部分"""
    logger = get_this_logger()
    index_arr = get_index_arr(img)
    img_mod = copy.deepcopy(img)

    # 使用C6,C7的位置估算颈椎区域
    cervical_labels = {
        "vertebrae C6": 19,
        "vertebrae C7": 18
    }
    for vertebra, label in cervical_labels.items():
        if np.any(img == label):
            cervical_points = index_arr[img == label]
            logger.info(f"Found {vertebra} at position {np.median(cervical_points, axis=0)}")
    
    # 使用L3-L5的位置估算腰椎区域
    lumbar_labels = {
        "vertebrae L3": 3,
        "vertebrae L4": 2,
        "vertebrae L5": 1
    }
    for vertebra, label in lumbar_labels.items():
        if np.any(img == label):
            lumbar_points = index_arr[img == label]
            logger.info(f"Found {vertebra} at position {np.median(lumbar_points, axis=0)}")

    return img_mod

def process_file(args):
    file_path, target_path = args
    file_name = file_path.split("/")[-1]
    nib_img = nib.load(file_path)
    img, header = nib_img.get_fdata().astype(np.uint8), nib_img.header
    
    logger = get_this_logger()
    logger.info(f"Processing {file_name}")

    try:
        # 1. 核心椎骨处理
        img = spine_adjacent_pairs(img, include_sacrum=True)
        logger.info(f"Completed spine processing")
        
        # 2. 位置检查
        img = bodypart_limitations(img)
        logger.info(f"Completed position check")
        
        # 3. 连通域处理
        img = supress_non_largest_components(img)
        logger.info(f"Completed connected components processing")
        
        postprcessed_img = nib.Nifti1Image(img.astype(np.uint8), affine=nib_img.affine, header=header)
        nib.save(postprcessed_img, target_path)
        
    except Exception as e:
        logger.error(f"Error processing {file_name}: {str(e)}")
        logger.error(traceback.format_exc())

def main(args):
    logger = get_this_logger()
    source_dir = "AbdomenAtlasDemo_preprocessed"
    target_dir = "AbdomenAtlasDemo_postprocessed"
    nof_jobs = args.nof_jobs

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for case_dir in os.listdir(source_dir):
        case_path = os.path.join(source_dir, case_dir)
        if os.path.isdir(case_path):
            for filename in os.listdir(case_path):
                if filename.endswith(".nii.gz"):
                    print(f"Processing {filename} in {case_dir}")
                    
                    filepath = os.path.join(case_path, filename)
                    target_case_dir = os.path.join(target_dir, case_dir)
                    if not os.path.exists(target_case_dir):
                        os.makedirs(target_case_dir)
                    
                    target_path = os.path.join(target_case_dir, f"postprocessing_{filename}")
                    process_file((filepath, target_path))
    
    print("处理完成")

if __name__ == "__main__":
    main(parser.parse_args())