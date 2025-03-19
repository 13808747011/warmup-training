:: filepath: yours\batch_process.bat
@echo off
python e:\zhuomian\SuPreM\submit\final_crop.py
python e:\zhuomian\SuPreM\submit\final_preprocess.py
python e:\zhuomian\SuPreM\submit\postprocessing_vertebrae.py
echo 批处理完成！
pause