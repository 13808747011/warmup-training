# SuPreM Batch Processing Guide

This project includes three main Python scripts and a batch file for cropping, preprocessing, and postprocessing medical imaging data.

## Folder Structure Requirements

Ensure the input folder structure is as follows:
```
AbdomenAtlasDemo
├── BDMAP_00000001
│   └── ct.nii.gz
├── BDMAP_00000002
│   └── ct.nii.gz
...
```
Each subfolder (e.g., `BDMAP_00000001`) contains one or more `.nii.gz` files.

## Steps to Use

### 1. Install Dependencies
Run the following command to install the required Python libraries:
```bash
pip install numpy nibabel cc3d scipy
```

### 2. Run the Batch File
Double-click `batch_process.bat` to start processing, or run the following commands in the command prompt:
```cmd
cd yourpath\submit
batch_process.bat
```

### 3. Output Folder Structure
After processing, you will get the following output folders:
```
AbdomenAtlasDemo_cropped
├── BDMAP_00000001
│   └── ct.nii.gz
├── BDMAP_00000002
│   └── ct.nii.gz
...

AbdomenAtlasDemo_preprocessed
├── BDMAP_00000001
│   └── ct.nii.gz
├── BDMAP_00000002
│   └── ct.nii.gz
...

AbdomenAtlasDemo_postprocessed
├── BDMAP_00000001
│   └── postprocessing_ct.nii.gz
├── BDMAP_00000002
│   └── postprocessing_ct.nii.gz
...
```

### 4. Script Descriptions

#### `final_crop.py`
- **Function**: Crops the input `.nii.gz` files and removes weak connections.
- **Input**: `AbdomenAtlasDemo`
- **Output**: `AbdomenAtlasDemo_cropped`

#### `final_preprocess.py`
- **Function**: Preprocesses the cropped files to handle big complex vertebrae.
- **Input**: `AbdomenAtlasDemo_cropped`
- **Output**: `AbdomenAtlasDemo_preprocessed`

#### `postprocessing_vertebrae.py`
- **Function**: Postprocesses the preprocessed files to adjust vertebrae connectivity.
- **Input**: `AbdomenAtlasDemo_preprocessed`
- **Output**: `AbdomenAtlasDemo_postprocessed`

### 5. Notes
- Ensure the input folder `AbdomenAtlasDemo` exists and follows the required structure.
- To process a different folder, replace `AbdomenAtlasDemo` with the new folder name while maintaining the same structure.
- If errors occur during execution, verify that all dependencies are installed and the input file format is correct.

