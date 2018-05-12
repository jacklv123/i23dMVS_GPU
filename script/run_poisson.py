#!/usr/bin/python
#! -*- encoding: utf-8 -*-

# Python implementation of the bash script written by Romuald Perrot
# Created by Baogui Yan
#
# this script is for easy use of I23DMVS
#
# usage : python run_poisson.py sfm_data_dir output_dir
#
# sfm_data_path is the input I23DSFM json file
# output_dir is where the project must be saved
#
# if output_dir is not present script will create it
#

# Indicate the I23D binary directory
I23D_BIN = "/home/yanbg/Project/i23dMVS_build_0524/bin/"
POISSON_BIN = "/home/yanbg/Project/PoissonReconTrim/Bin/Linux"

import commands
import os
import subprocess
import sys

if len(sys.argv) < 3:
    print ("Usage %s sfm_data_path output_dir" % sys.argv[0])
    sys.exit(1)

sfm_data_path = os.path.join(sys.argv[1], "sfm_data.json")
output_dir = sys.argv[2]
scene_path = os.path.join(output_dir, "scene.mvs")
working_dir = output_dir + "/intermediate"
use_poisson = False

print ("Using sfm data path  : ", sfm_data_path)
print ("      output_dir : ", output_dir)
print ("Reconstruct Mesh Use Poisson");

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(working_dir):
    os.mkdir(working_dir)

print ("1. Import Scene from I23dSFM")
pImport = subprocess.Popen( [os.path.join(I23D_BIN, "InterfaceI23dSFM"),  "-i", sfm_data_path, "-o", output_dir+"/scene.mvs", "-w", working_dir] )
pImport.wait()

print ("2. Dense Point-Cloud Reconstruction (optional)")
pPointRecon = subprocess.Popen( [os.path.join(I23D_BIN, "DensifyPointCloud"), "-i", output_dir+"/scene.mvs", "-w", working_dir, "--estimate-normals", "1", "--resolution-level", "1", "-v", "1"] )
pPointRecon.wait()

print ("3. Poisson Mesh Reconstruction")
pMeshRecon = subprocess.Popen( [os.path.join(POISSON_BIN, "PoissonRecon"), "--in", output_dir+"/scene_dense.ply", "--out", output_dir+"/scene_poisson.ply", "--depth", "11", "--trim", "--verbose"] )
pMeshRecon.wait()

print ("4. Mesh Texturing")
pTexture = subprocess.Popen( [os.path.join(I23D_BIN, "TextureMesh"), "-i", output_dir+"/scene.mvs", "-w", working_dir, "--mesh-file", output_dir+"/scene_poisson.ply", "--resolution-level", "2"] )
pTexture.wait()
