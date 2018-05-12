#include<stdio.h>
#include<cuda_runtime.h>
#include<cuda.h>
#include"utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include"FuseDepthMaps_gpu.h"
#include"EstimateDepthMaps_gpu.h"


FuseDepthMaps_gpu fuser;
EstimateDepthMaps_gpu Estimater;

void FuseDepthMaps_GPU( int * connection_data,
			int n,
			int height,
			int width,
			float * h_images_K,
			float * h_images_R,
			float * h_images_C,

			float * h_depthmap,
			float * h_confmap,
			int * h_neighbors,
			int * h_neighbors_begin,
			int * h_neighbors_end,
			int nPointsEstimate,
			float * h_points,
			float * h_weight,
			int * h_view,
			int * h_pointsinfo
){
	fuser.Malloc_memory(n,nPointsEstimate,height,width);

	fuser.copy_input_data(
			connection_data,
			n,
			h_images_K,
			h_images_R,
			h_images_C,
			h_depthmap,
			h_confmap,
			h_neighbors,
			h_neighbors_begin,
			h_neighbors_end
	);

	fuser.FuseDepthMaps(n);
	fuser.Carry_Out(
		h_points,
		h_weight,
		h_view,
		h_pointsinfo,
		n,
		nPointsEstimate
	);

	fuser.MemoryFree(n,nPointsEstimate);
	//fuser.hello();
	//for (int i=0;i<n;i++) printf("%d\n",value[i]);
}

void InitialAndSendImageData(int n,
		   int height,
		   int width,
		   float * h_images,
		   float * h_images_R,
		   float * h_images_K,
		   float * h_images_C){

	Estimater.ComputeCudaConfig(n,height,width);

	Estimater.Malloc_memory(n,height,width);

	Estimater.SendImageData(h_images,h_images_R,h_images_K,h_images_C);
}
/*
void SendRefData(int idxImage,
		 //float * h_depthmap,
		 //float * h_confmap,
		 //float * h_normalmap
		thrust :: host_vector <float > h_depthmap,
		thrust :: host_vector <float > h_confmap,
		thrust :: host_vector <float > h_normalmap
){

	Estimater.SendRefData(idxImage,h_depthmap,h_confmap,h_normalmap);
}
*/
void EstimateDepthMap_GPU(
			int ref,
			float dMin,
			float dMax,
			float * h_depthmap,
			float * h_confmap,
			float * h_normalmap,
			int * neighbors
){
	Estimater.EstimateDepthMaps(ref,dMin,dMax,h_depthmap,h_confmap,h_normalmap,neighbors);
}
/*
void CarryOutData(int idxImage,
		 //float * h_depthmap,
		 //float * h_confmap,
		 //float * h_normalmap
		thrust :: host_vector <float > h_depthmap,
		thrust :: host_vector <float > h_confmap,
		thrust :: host_vector <float > h_normalmap
){
	Estimater.CarryOutData(idxImage,h_depthmap,h_confmap,h_normalmap);
}
*/
void MemoryFree(){
	Estimater.MemoryFree();
}