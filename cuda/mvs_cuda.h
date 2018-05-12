#include "timer.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/////////////////////////////////////////

void FuseDepthMaps_GPU( int * h_connection_data,
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

);
void InitialAndSendImageData(int n,
		   int height,
		   int width,
		   float * h_images,
		   float * h_images_R,
		   float * h_images_K,
		   float * h_images_C
);
/*
void SendRefData(int idxImage,
		 //float * h_depthmap,
		 //float * h_confmap,
		 //float * h_normalmap
		thrust :: host_vector <float > h_depthmap,
		thrust :: host_vector <float > h_confmap,
		thrust :: host_vector <float > h_normalmap

);
*/
void EstimateDepthMap_GPU(
			int ref,
			float dMin,
			float dMax,
			float * d_depthmap,
			float * d_confmap,
			float * d_normalmap,
			int * neighbors
			);
/*
void CarryOutData(int idxImage,
		 //float * h_depthmap,
		 //float * h_confmap,
		 //float * h_normalmap
		thrust :: host_vector <float > h_depthmap,
		thrust :: host_vector <float > h_confmap,
		thrust :: host_vector <float > h_normalmap

);
*/
void MemoryFree();