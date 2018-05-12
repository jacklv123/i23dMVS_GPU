# How to run I23d
1. Build i23d follow the steps in BUILD.md
2. Modify run_i23d.py and run_poisson.py to specify the i23d bin directory
4. python i23dSFM_build/software/SfM/SfM_SequentialPipeline.py image_dir output_dir
5. python run_i23d.py/run_poisson.py sfm_data_dir output_dir
# For Viwo
6. run TextureMeshViwo in i23dMVS_build/bin, take care of the parameters

