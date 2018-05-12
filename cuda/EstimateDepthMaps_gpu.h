//this class is aim to estimate depthmaps
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <curand.h>
#include <curand_kernel.h>
#include<cmath>

#define PropagationBlockCol 32
#define RandomRefinementBlockCol 16
#define RandomRefinementBlockRow 32

class EstimateDepthMaps_gpu{
public:
	__host__ __device__ EstimateDepthMaps_gpu();
	__host__ __device__ ~EstimateDepthMaps_gpu();

	void Malloc_memory(int n,int height,int width);

	void SendImageData(
		float *h_images,
		float *h_images_R,
		float *h_images_K,
		float * h_images_C);

	/*void SendRefData(
		int idxImage,
		//float * h_depthmap,
		//float * h_confmap,
		//float * h_normalmap
		thrust :: host_vector <float > h_depthmap,
		thrust :: host_vector <float > h_confmap,
		thrust :: host_vector <float > h_normalmap
		);
	*/
	
	void EstimateDepthMaps(
							int ref,
							float dMin,
							float dMax,
							float * h_depthmap,
							float * h_confmap,
							float * h_normalmap,
							int * neighbors
						);
	/*
	void CarryOutData(
		int idxImage,
		//float * h_depthmap,
		//float * h_confmap,
		//float * h_normalmap
		thrust :: host_vector <float > &h_depthmap,
		thrust :: host_vector <float > &h_confmap,
		thrust :: host_vector <float > &h_normalmap
		);
	*/
	void ComputeCudaConfig(int n,int height_,int width_);
	void MemoryFree();
	
	// Dimensions for sweeping from top to bottom, i.e. one thread per column.
	//image parallel
  	dim3 ColParallelBlockSize;
  	dim3 ColParallelGridSize;
 	dim3 PixelParallelBlockSize;
	dim3 PixelParallelGridSize;
	dim3 randstatesBlockSize;
	dim3 randstatesGridSize;
	
	int n;
	int height;
	int width;
	int nSizeHalfWindow;
	int nSizeWindow;

	//float * d_confmap;//n*h*w
	//float * d_depthmap;//n*h*w
	float * d_images;//n*h*w
	//float * d_normalmap;//n*h*w*3
	float * d_images_R;//n*9
	float * d_images_K;//n*9
	float * d_images_C;//n*3
	float * h_ref_K;
	float * h_ref_inv_K;
	//float * ref_K;
	float * ref_inv_K;
	curandState * randstates;
};
// Calibration of reference image as {fx, cx, fy, cy}.
//__constant__ float ref_K[4];
// Calibration of reference image as {1/fx, -cx/fx, 1/fy, -cy/fy}.
//__constant__ float ref_inv_K[4];
__constant__ float scaleRanges[12];
	
namespace EstimateDepthMapsKernel {
	
	//内参矩阵
	//|fx    u0| 
	//|   fy v0|
	//|       1|
	//旋转矩阵
	//|0 1 2|
	//|3 4 5|
	//|6 7 8|
	/*// Calibration of reference image as {fx, cx, fy, cy}.
	invK(0,0) = REAL(1)/K(0,0);
	invK(1,1) = REAL(1)/K(1,1);
	invK(0,2) = -K(0,2)*invK(0,0);
	invK(1,2) = -K(1,2)*invK(1,1);
	*/
	__device__  float sample(float * image, 
							 int src,
							 float2 pt,
							 int height,
							 int width
							 ){
		
		const int lx=(int)pt.x;
		const int ly=(int)pt.y;
		if (lx>=1&&lx+1<width&&ly>=1&&ly+1<height) {} else return -1.0;
		
		const float x=pt.x-lx, x1=1.0-x;
		const float y=pt.y-ly, y1=1.0-y;
				
		int base=src*width*height;
		
		return (
				(image[base+ly*width+lx]*x1+image[base+ly*width+lx+1]*x)*y1    +
				(image[base+(ly+1)*width+lx]*x1+image[base+(ly+1)*width+lx+1]*x)*y
			   );
	}
	/*
	__device__ inline float PropagateDepth(const float depth1,
                                       const float3 normal1, 
									   const float row1,
                                       const float row2,
									   const int ref
									   ) {
	  // Point along first viewing ray.
	  const float x1 = depth1 * (ref_inv_K[2] * row1 + ref_inv_K[3]);
	  const float y1 = depth1;
	  // Point on plane defined by point along first viewing ray and plane normal1.
	  const float x2 = x1 + normal1.z;
	  const float y2 = y1 - normal1.y;

	  // Origin of second viewing ray.
	  // const float x3 = 0.0f;
	  // const float y3 = 0.0f;
	  // Point on second viewing ray.
	  const float x4 = ref_inv_K[2] * row2 + ref_inv_K[3];
	  // const float y4 = 1.0f;

	  // Intersection of the lines ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4)).
	  const float denom = x2 - x1 + x4 * (y1 - y2);
	  const float kEps = 1e-5f;
	  if (abs(denom) < kEps) {
		return depth1;
	  }
	  const float nom = y1 * x2 - x1 * y2;
	  return nom / denom;
	}
	*/
	__device__ inline float3 Mat33DotVec3(const float mat[9], const float3 vec
                                    ) {
		float3 result;
		result.x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
		result.y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
		result.z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
		return result;
	}
	__device__ inline float DotProduct3(float3 vec1, float3 vec2) {
		return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
	}
	__device__ inline float GenerateRandomDepth0(const float depth_min,
												const float depth_max,
												curandState* rand_state) {
	  return curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
	}

	__device__ inline float3 GenerateRandomNormal(
												int ref,
												const int row, 
												const int col,
												curandState* rand_state,
												float * ref_inv_K
												) {
	  // Unbiased sampling of normal, according to George Marsaglia, "Choosing a
	  // Point from the Surface of a Sphere", 1972.
	  float3 normal;
	  float v1 = 0.0f;
	  float v2 = 0.0f;
	  float s = 2.0f;
	  while (s >= 1.0f) {
		v1 = 2.0f * curand_uniform(rand_state) - 1.0f;
		v2 = 2.0f * curand_uniform(rand_state) - 1.0f;
		s = v1 * v1 + v2 * v2;
	  }

	  const float s_norm = sqrt(1.0f - s);
	  normal.x = 2.0f * v1 * s_norm;
	  normal.y = 2.0f * v2 * s_norm;
	  normal.z = 1.0f - 2.0f * s;

	  // Make sure normal is looking away from camera.
	  float3 view_ray;
	  view_ray.x=ref_inv_K[ref*4+0] * col + ref_inv_K[ref*4+1];
	  view_ray.y=ref_inv_K[ref*4+2] * row + ref_inv_K[ref*4+3];
	  view_ray.z=1.0f;
	  
	  if (DotProduct3(normal, view_ray) > 0) {
		normal.x = -normal.x;
		normal.y = -normal.y;
		normal.z = -normal.z;
	  }
	  return normal;
	}
	__device__ inline float CLAMP(float v,float c0,float c1){
		return fminf(fmaxf(v,c0),c1);
	}
	
	__device__ inline float GenerateRandomDepth(const float depth_min,
                                            const float depth_max,
                                            curandState* rand_state,
											float depth,
											float range) {
		
		float ans=depth-range+curand_uniform(rand_state)*(2*range);
		
		return ans;
	}

	__device__ inline float3 PerturbNormal(
									const int ref,
									const int row, 
									const int col,
                                    const float perturbation,
                                    const float3 normal,
                                    curandState* rand_state,
									float * ref_inv_K
                                     ) {
		float3 perturbed_normal;
		// Perturbation rotation angles.
		const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
		const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
		const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

		const float sin_a1 = sin(a1);
		const float sin_a2 = sin(a2);
		const float sin_a3 = sin(a3);
		const float cos_a1 = cos(a1);
		const float cos_a2 = cos(a2);
		const float cos_a3 = cos(a3);

		// R = Rx * Ry * Rz
		float R[9];
		R[0] = cos_a2 * cos_a3;
		R[1] = -cos_a2 * sin_a3;
		R[2] = sin_a2;
		R[3] = cos_a1 * sin_a3 + cos_a3 * sin_a1 * sin_a2;
		R[4] = cos_a1 * cos_a3 - sin_a1 * sin_a2 * sin_a3;
		R[5] = -cos_a2 * sin_a1;
		R[6] = sin_a1 * sin_a3 - cos_a1 * cos_a3 * sin_a2;
		R[7] = cos_a3 * sin_a1 + cos_a1 * sin_a2 * sin_a3;
		R[8] = cos_a1 * cos_a2;

		// Perturb the normal vector.
		perturbed_normal=Mat33DotVec3(R, normal);

		// Make sure the perturbed normal is still looking in the same direction as
		// the viewing direction.
		float3 view_ray ;
		view_ray.x= ref_inv_K[ref*4+0] * col + ref_inv_K[ref*4+1];
		view_ray.y= ref_inv_K[ref*4+2] * row + ref_inv_K[ref*4+3]; 
		view_ray.z= 1.0f;
		
		if (DotProduct3(perturbed_normal, view_ray) >= 0.0f) {
		perturbed_normal.x = normal.x;
		perturbed_normal.y = normal.y;
		perturbed_normal.z = normal.z;
		}

		// Make sure normal has unit norm.
		const float inv_norm = rsqrt(DotProduct3(perturbed_normal, perturbed_normal));
		perturbed_normal.x *= inv_norm;
		perturbed_normal.y *= inv_norm;
		perturbed_normal.z *= inv_norm;
		return perturbed_normal;
	}
	__device__ inline void Mat3X3Inv(int ref,float* d_images_R,float Inv[9]){
		float a1=d_images_R[ref*9+0];
		float b1=d_images_R[ref*9+1];
		float c1=d_images_R[ref*9+2];
		float a2=d_images_R[ref*9+3];
		float b2=d_images_R[ref*9+4];
		float c2=d_images_R[ref*9+5];
		float a3=d_images_R[ref*9+6];
		float b3=d_images_R[ref*9+7];
		float c3=d_images_R[ref*9+8];
		float co=1.0/( a1*(b2*c3-c2*b3)-a2*(b1*c3-c1*b3)+a3*(b1*c2-c1*b2) );

		Inv[0]=(b2*c3-c2*b3)*co;
		Inv[1]=(c1*b3-b1*c3)*co;
		Inv[2]=(b1*c2-c1*b2)*co;
		Inv[3]=(c2*a3-a2*c3)*co;
		Inv[4]=(a1*c3-c1*a3)*co;
		Inv[5]=(a2*c1-a1*c2)*co;
		Inv[6]=(a2*b3-b2*a3)*co;
		Inv[7]=(b1*a3-a1*b3)*co;
		Inv[8]=(a1*b2-a2*b1)*co;
	}
	__device__ inline void MatMul3x3(float a[9],float b[9],float c[9]){
		for (int i=0;i<9;i++) c[i]=0;
		for (int i=0;i<3;i++)
			for (int j=0;j<3;j++)
				for (int k=0;k<3;k++){
					int cid=i*3+j;
					int aid=i*3+k;
					int bid=k*3+j;
					c[cid]=c[cid]+a[aid]*b[bid];
				}
	}
	__device__ inline void ComposeHomography(
											const int ref,
											const int src, 
											const int row,
											const int col, 
											const float depth,
											const float3 normal, 
											float H[9],
											const EstimateDepthMaps_gpu data) {
		// Calibration of source image
		float K_src[9];
		float K_ref_inv[9];
		for (int i=0;i<9;i++){
			K_src[i]=data.d_images_K[src*9+i];
		}
		Mat3X3Inv(ref,data.d_images_K,K_ref_inv);
		
		/*
		K[0]=data.d_images_K[src*9+0];
		K[1]=data.d_images_K[src*9+2];
		K[2]=data.d_images_K[src*9+4];
		K[3]=data.d_images_K[src*9+5];
		*/
		
		
		// Relative rotation between reference and source image.
		float R_src[9];
		float R_ref_inv[9];
		float R[9];
		for (int i = 0; i < 9; ++i) {
			R_src[i] = data.d_images_R[src*9+i];
		}
		Mat3X3Inv(ref,data.d_images_R,R_ref_inv);
		MatMul3x3(R_src,R_ref_inv,R);
		// Relative translation between reference and source image.
		float T0[3],T[3];
		for (int i = 0; i < 3; ++i) {
			T0[i] = data.d_images_C[ref*3+i]-data.d_images_C[src*3+i];
		}
		T[0]=T0[0]*R[0]+T0[1]*R[1]+T0[2]*R[2];
		T[1]=T0[0]*R[3]+T0[1]*R[4]+T0[2]*R[5];
		T[2]=T0[0]*R[6]+T0[1]*R[7]+T0[2]*R[8];
		
		// Distance to the plane.
		const float dist =
		  depth * (normal.x * (data.ref_inv_K[ref*4+0] * col + data.ref_inv_K[ref*4+1]) +
				   normal.y * (data.ref_inv_K[ref*4+2] * row + data.ref_inv_K[ref*4+3]) + normal.z);
		const float inv_dist = 1.0f / dist;
		
		float inv_dist_N[3];
		inv_dist_N[0] = inv_dist * normal.x;
		inv_dist_N[1] = inv_dist * normal.y;
		inv_dist_N[2] = inv_dist * normal.z;
		
		float t[9];
		for (int i=0;i<3;i++)
			for (int j=0;j<3;j++){
				t[i*3+j]=T[i]*inv_dist_N[j];
			}
		float p[9];
		MatMul3x3(R_src,t,p);
		for (int i=0;i<9;i++) R[i]+=p[i];
		MatMul3x3(K_src,R,t);
		MatMul3x3(t,K_ref_inv,H);
		// Homography as H = K * (R - T * n' / d) * Kref^-1.
		/*
		H[0] = data.ref_inv_K[ref*4+0] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
							 K[1] * (R[6] + inv_dist_N0 * T[2]));
		H[1] = data.ref_inv_K[ref*4+2] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
							 K[1] * (R[7] + inv_dist_N1 * T[2]));
		H[2] = K[0] * (R[2] + inv_dist_N2 * T[0]) +
			 K[1] * (R[8] + inv_dist_N2 * T[2]) +
			 data.ref_inv_K[ref*4+1] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
							 K[1] * (R[6] + inv_dist_N0 * T[2])) +
			 data.ref_inv_K[ref*4+3] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
							 K[1] * (R[7] + inv_dist_N1 * T[2]));
		H[3] = data.ref_inv_K[ref*4+0] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
							 K[3] * (R[6] + inv_dist_N0 * T[2]));
		H[4] = data.ref_inv_K[ref*4+2] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
							 K[3] * (R[7] + inv_dist_N1 * T[2]));
		H[5] = K[2] * (R[5] + inv_dist_N2 * T[1]) +
			 K[3] * (R[8] + inv_dist_N2 * T[2]) +
			 data.ref_inv_K[ref*4+1] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
							 K[3] * (R[6] + inv_dist_N0 * T[2])) +
			 data.ref_inv_K[ref*4+3] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
							 K[3] * (R[7] + inv_dist_N1 * T[2]));
		H[6] = data.ref_inv_K[ref*4+0] * (R[6] + inv_dist_N0 * T[2]);
		H[7] = data.ref_inv_K[ref*4+2] * (R[7] + inv_dist_N1 * T[2]);
		H[8] = R[8] + data.ref_inv_K[ref*4+1] * (R[6] + inv_dist_N0 * T[2]) +
			 data.ref_inv_K[ref*4+3] * (R[7] + inv_dist_N1 * T[2]) + inv_dist_N2 * T[2];
		*/
	}
	// The return values is 1 - NCC, so the range is [0, 2], the smaller the
	// value, the better the color consistency.
	struct PhotoConsistencyCostComputer {
	  EstimateDepthMaps_gpu data;
	  // Identifier of source image.
	  int ref=0;
	  int kWindowSize=7;
	  int PatchSize=49;
	  int src_image_id = -1;

	  // Center position of patch in reference image.
	  int row = -1;
	  int col = -1;

	  // Depth and normal for which to warp patch.
	  float depth = 0.0f;
	  float3 normal;

	  // Dimensions of reference image.
	  int width = 0;
	  int height = 0;

	  __device__ inline float Compute() const {
		const float kMaxCost = 2.0f;
		const int kWindowRadius = kWindowSize / 2;
		
		const int row_start = row - kWindowRadius;
		const int col_start = col - kWindowRadius;
		const int row_end = row + kWindowRadius;
		const int col_end = col + kWindowRadius;

		if (row_start < 0 || col_start < 0 || row_end >= height ||
			col_end >= width) {
		  return kMaxCost;
		}

		float tform[9];
														
		ComposeHomography(
				ref,
				src_image_id, //const int src, 
				row, //const int row,
				col, //const int col, 
				depth, //const float depth,
				normal, //const float3 normal, 
				tform,//float H[9]
				data //const EstimateDepthMaps_gpu data
				);

		float col_src = tform[0] * col_start + tform[1] * row_start + tform[2];
		float row_src = tform[3] * col_start + tform[4] * row_start + tform[5];
		float z = tform[6] * col_start + tform[7] * row_start + tform[8];
		float base_col_src = col_src;
		float base_row_src = row_src;
		float base_z = z;		
		
		float sigmaQ=0.0;
		float sigmaH=0.0;
		float sigmaQQ=0.0;
		float sigmaHH=0.0;
		float sigmaQH=0.0;
		
		for (int r=-kWindowRadius+row;r<=row+kWindowRadius;r++){
			for (int c=-kWindowRadius+col;c<=col+kWindowRadius;c++){
				if (!(r>=0&&r<height)||!(c>=0&&c<width)) {
					printf("%d %d\n",r,c);
					return kMaxCost;
				}

				const float q=data.d_images[ref*width*height+r*width+c];
				
				const float inv_z = 1.0f / z;
				const float norm_col_src = inv_z * col_src + 0.5f;
				const float norm_row_src = inv_z * row_src + 0.5f;
				//printf("%lf %lf\n",norm_col_src,norm_row_src)
				const float H=sample(
							data.d_images,//float * image, 
							src_image_id, //int src,
							make_float2(norm_col_src,norm_row_src),//float2 pt,
							height,
							width
							);
				
				if (H<0) return kMaxCost;
				
				sigmaQ+=q;
				sigmaQQ+=q*q;
				
				sigmaH+=H;
				sigmaHH+=H*H;
				
				sigmaQH+=q*H;
				
				col_src += tform[0];
				row_src += tform[3];
				z += tform[6];
			}
			
			base_col_src += tform[1];
			base_row_src += tform[4];
			base_z += tform[7];

			col_src = base_col_src;
			row_src = base_row_src;
			z = base_z;
		}
		float nrm=(sigmaQQ-sigmaQ*sigmaQ/PatchSize) *  (sigmaHH-sigmaH*sigmaH/PatchSize);
		if (fabsf(nrm)<0.000000001) return kMaxCost;
		
		float ans=( sigmaQH-sigmaQ*sigmaH/PatchSize)/sqrt(nrm);
		ans=CLAMP(ans,-1.0,1.0);
		//printf("%lf\n",1-ans);
		return 1-ans;
		
	  }
	};
	 
	__device__  inline unsigned DecodeScoreScale(float& score) {
		unsigned invScaleRange=int(score);
		score = (score-1.0*invScaleRange)*10.0;
		return invScaleRange;
	}
	__device__ inline float EncodeScoreScale(float score, unsigned invScaleRange=0) {
		return score*0.1f+(float)invScaleRange;
	}
	
	__device__ inline float ComputeAngle(const float3 V1, const float3 V2) {
	// compute the angle between the rays
		float ans=(V1.x*V2.x+V1.y*V2.y+V1.z*V2.z)/sqrt((V1.x*V1.x+V1.y*V1.y+V1.z*V1.z)*(V2.x*V2.x+V2.y*V2.y+V2.z*V2.z));
		ans=CLAMP(ans,-1.0,1.0);
	} //
	/*
	MDEFVAR_OPTDENSE_float(fNCCThresholdKeep, "NCC Threshold Keep", "Maximum 1-NCC score accepted for a match", "0.5", "0.3")
	MDEFVAR_OPTDENSE_float(fNCCThresholdRefine, "NCC Threshold Refine", "1-NCC score under which a match is not refined anymore", "0.03")

	thConfSmall(OPTDENSE::fNCCThresholdKeep*0.25f),
	thConfBig(OPTDENSE::fNCCThresholdKeep*0.5f),
	thConfIgnore(OPTDENSE::fNCCThresholdKeep*1.5f)
	
	float& conf = confMap0(x);
	unsigned invScaleRange(DecodeScoreScale(conf));
	
	const float newconf(ScorePixel(neighbor.depth, neighbor.normal));
	if (conf > newconf) {
		conf = newconf;
		depth = neighbor.depth;
		normal = neighbor.normal;
		invScaleRange = (ninvScaleRange>1 ? ninvScaleRange-1 : ninvScaleRange);
	}
	
	conf = EncodeScoreScale(conf, invScaleRange);
	*/
	__global__ void initialRandStates(
		curandState * randstates,
		unsigned int seed,
		int width
	){
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		int col = blockIdx.y * blockDim.y + threadIdx.y;
		curand_init(
			seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
            idx*width+col, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
            0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
            &randstates[idx*width+col]
		);
	}
	
	__global__ void initialConfMap(
		EstimateDepthMaps_gpu data,
		int ref,
		float dMin,
		float dMax,
		float * d_depthmap,
		float * d_confmap,
		float * d_normalmap,
		int * d_neighbors
	){
		int height=data.height;
		int width=data.width;
		int n=data.n;

		int row = blockIdx.x * blockDim.x + threadIdx.x;
		int col = blockIdx.y * blockDim.y + threadIdx.y;
		if (col>=data.nSizeHalfWindow&&col<width-data.nSizeHalfWindow){} else return ;
		if (row>=data.nSizeHalfWindow&&row<height-data.nSizeHalfWindow) {} else return ;
				
		struct ParamState {
			float depth; 
			float3 normal;
			
		};
		ParamState curr_param_state;
		
		curr_param_state.depth=d_depthmap[row*width+col];
		curr_param_state.normal=make_float3(
			d_normalmap[(row*width+col)*3+0],
			d_normalmap[(row*width+col)*3+1],
			d_normalmap[(row*width+col)*3+2]
		); 
		PhotoConsistencyCostComputer pcc_computer;
		
		pcc_computer.kWindowSize=data.nSizeWindow;
		pcc_computer.data=data;
		pcc_computer.PatchSize=data.nSizeWindow*data.nSizeWindow;
		pcc_computer.height=height;
		pcc_computer.width=width;
		pcc_computer.ref=ref;
		pcc_computer.depth=curr_param_state.depth;
		pcc_computer.normal=curr_param_state.normal;
		pcc_computer.row=row;
		pcc_computer.col=col;
		
		int nVilidScore;
		float avg_score;
		nVilidScore=0;
		avg_score=0;
		for (int i=2;i<d_neighbors[0];i++){
			pcc_computer.src_image_id=d_neighbors[i];
			float score=pcc_computer.Compute();
			if (score<2.0){
				
				nVilidScore++;
				avg_score+=score;
			}
		}
		float CurScore;
		if (nVilidScore!=0) CurScore=avg_score/nVilidScore; else  CurScore=2.0;
		unsigned invScaleRange=DecodeScoreScale(d_confmap[row*width+col]);
		printf("%lf %lf %d %lf %d %d\n",d_confmap[row*width+col],CurScore,nVilidScore,avg_score,row,col);
		d_confmap[row*width+col]=EncodeScoreScale(CurScore,invScaleRange);
		
		//if (CurScore<2) printf("%.10lf\n",CurScore);
	}
	__global__ void initialDepthAndNormalMap(
		EstimateDepthMaps_gpu data,
		int ref,
		float dMin,
		float dMax,
		float * d_depthmap,
		float * d_confmap,
		float * d_normalmap
	){
		int height=data.height;
		int width=data.width;

		int col = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (col>=0&&col<width){} else return ;
		
		curandState randstate=data.randstates[ref*width+col];
		
		struct ParamState {
			float depth; 
			float3 normal;
			
		};
		ParamState curr_param_state;
		
		for (int row=0;row<height;row++){
			if (fabsf(d_depthmap[row*width+col])>0.000000001){
				curr_param_state.depth=d_depthmap[row*width+col];
				curr_param_state.normal.x=d_normalmap[(row*width+col)*3+0];
				curr_param_state.normal.y=d_normalmap[(row*width+col)*3+1];
				curr_param_state.normal.z=d_normalmap[(row*width+col)*3+2];
			} else {
				curr_param_state.depth=GenerateRandomDepth0(
					dMin,
					dMax,
					&randstate
				);
				//printf("%lf %lf %lf\n",dMin,dMax,curr_param_state.depth);
					
				curr_param_state.normal=GenerateRandomNormal(
					ref,
					row,
					col,
					&randstate,
					data.ref_inv_K
				);									
			}
			//if (curr_param_state.depth>=dMin&&curr_param_state.depth<=dMax){} else 	
				//printf("%lf %lf %lf\n",dMin,dMax,curr_param_state.depth);
			//printf("%d %lf\n",col,curr_param_state.depth);	
			d_depthmap[row*width+col]=curr_param_state.depth;
			d_normalmap[(row*width+col)*3+0]=curr_param_state.normal.x;
			d_normalmap[(row*width+col)*3+1]=curr_param_state.normal.y;
			d_normalmap[(row*width+col)*3+2]=curr_param_state.normal.z;
		}
		
		data.randstates[ref*width+col]=randstate;
	}
	
	__global__ void PropagationKernel(
	EstimateDepthMaps_gpu data,
	int ref,//当前的相机编号
	int dir,//0：正向，1：从后往前
	float dMax,
	float dMin,
	float * d_depthmap,
	float * d_confmap,
	float * d_normalmap,
	int * d_neighbors
	)
	{
		int height=data.height;
		int width=data.width;
		int n=data.n;
		int col = blockIdx.x * blockDim.x + threadIdx.x;//需要传播的列
		if (col>=data.nSizeHalfWindow && col<width-data.nSizeHalfWindow ) {} else return ;
		curandState randstate=data.randstates[ref*width+col];
						
		PhotoConsistencyCostComputer pcc_computer;
		pcc_computer.kWindowSize=data.nSizeWindow;
		pcc_computer.data=data;
		pcc_computer.height=data.height;
		pcc_computer.width=data.width;
		pcc_computer.ref=ref;
		pcc_computer.col=col;
		
		struct ParamState {
			float depth; 
			float3 normal;
			
		};
		
		ParamState prev_param_state;
		ParamState curr_param_state;
		ParamState rand_param_state;
		prev_param_state.depth=d_depthmap[col];
		prev_param_state.normal=make_float3(
				d_normalmap[col*3+0],
				d_normalmap[col*3+1],
				d_normalmap[col*3+2]
				);
		
		for (int Row=data.nSizeHalfWindow;Row<height-data.nSizeHalfWindow;Row++){
			int row;
			if (dir) row=height-Row-1; else row=Row;
			int prerow=row + (dir?1:-1);
			
			curr_param_state.depth=d_depthmap[row*width+col];
			curr_param_state.normal=make_float3(
				d_normalmap[(row*width+col)*3+0],
				d_normalmap[(row*width+col)*3+1],
				d_normalmap[(row*width+col)*3+2]
			);
			 

			float conf=d_confmap[row*width+col];
			unsigned invScaleRange=DecodeScoreScale(conf);
			
			/*
			if (prerow>=0&&prerow<height)
				prev_param_state.depth = PropagateDepth(
											prev_param_state.depth, 
											prev_param_state.normal,
											prerow, 
											row,
											ref
											);
			*/
			if (invScaleRange<=2||conf>0.03){
				//up to bottom
				
				int nVilidScore=0;
				float avg_score=0;
	
				pcc_computer.row=row;
				
				float nconf=d_confmap[prerow*width+col];
				unsigned ninvScaleRange=DecodeScoreScale(nconf);
				
				//if (nconf >= thConfIgnore) 没加
				pcc_computer.depth=prev_param_state.depth;
				pcc_computer.normal=prev_param_state.normal;
				
				for (int i=2;i<d_neighbors[0];i++){
					pcc_computer.src_image_id=d_neighbors[i];
					float score=pcc_computer.Compute();
					if (score<2.0){
						nVilidScore++;
						avg_score+=score;
					}
				}
				float PreScore;
				if (nVilidScore!=0) PreScore=avg_score/nVilidScore; else PreScore=2.0;

				if (PreScore<conf){
					curr_param_state=prev_param_state;
					conf=PreScore;
					invScaleRange=(ninvScaleRange>1 ? ninvScaleRange-1 : ninvScaleRange);
				}
				
				float depthRange=curr_param_state.depth*0.01;
				
				if (invScaleRange > 2) invScaleRange = 2; else 
				if (invScaleRange == 0) {
					if (conf <=0.125 )//thConfSmall
						invScaleRange = 1;
					else 
					if (conf <= 0.25)//thConfBig
						depthRange *= 0.5f;
				}
				float scaleRange=scaleRanges[invScaleRange];
				
				//printf("%lf %lf %lf\n",dMin,dMax,curr_param_state.depth);
				/*
				for (int iter=0,enditer=6-invScaleRange;iter<enditer;iter++){
					rand_param_state.depth=GenerateRandomDepth(
					dMin,
					dMax,
					&randstate,
					curr_param_state.depth,
					depthRange*scaleRange);
					//printf("%lf %lf %lf %lf\n",dMin,dMax,rand_param_state.depth,curr_param_state.depth);
					if (rand_param_state.depth>=dMin&&rand_param_state.depth<=dMax){} else continue;
					
					rand_param_state.normal=PerturbNormal(
					ref,
					row,
					col,
                    3.1415926*0.02*scaleRange,
                    curr_param_state.normal,
                    &randstate,
					data.ref_inv_K
                    );
					if (curr_param_state.normal.z>=0) continue;
					
					nVilidScore=0;
					avg_score=0;	
					//printf("%lf %lf\n",curr_param_state.depth,rand_param_state.depth);
					pcc_computer.depth=rand_param_state.depth;
					pcc_computer.normal=rand_param_state.normal;
				
					for (int i=2;i<d_neighbors[0];i++){
						pcc_computer.src_image_id=d_neighbors[i];
						float score=pcc_computer.Compute();
						if (score<2.0){
							nVilidScore++;
							avg_score+=score;
						}
					}
					float RandScore;
					if (nVilidScore!=0) RandScore=avg_score/nVilidScore; else RandScore=2.0;
					
					//printf("%lf %lf\n",conf,RandScore);					

					if (conf>RandScore){
						conf=RandScore;
						curr_param_state=rand_param_state;
						scaleRange*=0.5f;
						++invScaleRange;
					}
				}
				*/
				
			}
			
			//printf("%d\n",invScaleRange);
			prev_param_state=curr_param_state;
			d_confmap[row*width+col]=EncodeScoreScale(conf, invScaleRange);
			
			d_depthmap[row*width+col]=curr_param_state.depth;
			d_normalmap[(row*width+col)*3+0]=curr_param_state.normal.x;
			d_normalmap[(row*width+col)*3+1]=curr_param_state.normal.y;
			d_normalmap[(row*width+col)*3+2]=curr_param_state.normal.z;
		}
		data.randstates[ref*width+col]=randstate;
	}

}

__host__ __device__ EstimateDepthMaps_gpu::EstimateDepthMaps_gpu(){

}
__host__ __device__ EstimateDepthMaps_gpu::~EstimateDepthMaps_gpu(){

}

void EstimateDepthMaps_gpu::Malloc_memory(int N,int Height,int Width){
	n=N;
	height=Height;
	width=Width;
	checkCudaErrors(cudaMalloc(&randstates,   sizeof(curandState) * n*width));
	EstimateDepthMapsKernel::initialRandStates<<<randstatesGridSize,randstatesBlockSize>>>(randstates,time(0),width);
	
	//checkCudaErrors(cudaMalloc(&d_confmap,   sizeof(float) * n*height*width));
	//checkCudaErrors(cudaMalloc(&d_depthmap,   sizeof(float) * n*height*width));
	//checkCudaErrors(cudaMalloc(&d_normalmap,   sizeof(float) * 3*n*height*width));
	checkCudaErrors(cudaMalloc(&d_images,   sizeof(float) * n*height*width));
	checkCudaErrors(cudaMalloc(&d_images_R,   sizeof(float) * n*9));
	checkCudaErrors(cudaMalloc(&d_images_K,   sizeof(float) * n*9));
	checkCudaErrors(cudaMalloc(&d_images_C,   sizeof(float) * n*3));
	
	//checkCudaErrors(cudaMalloc(&ref_K,   sizeof(float) * n*4));
	checkCudaErrors(cudaMalloc(&ref_inv_K,   sizeof(float) * n*4));

	//depthmap.resize(n*height*width);
	//confmap.resize(n*height*width);
	//normalmap.resize(n*height*width*3);
	
	printf("Malloc_memory is completed\n");
}
void EstimateDepthMaps_gpu::SendImageData(
						  float *h_images,
						  float *h_images_R,
						  float *h_images_K,
						  float * h_images_C
						  ){
	checkCudaErrors(cudaMemcpy(d_images, h_images, sizeof(float) * n*height*width,
                              cudaMemcpyHostToDevice));
	//camera
	checkCudaErrors(cudaMemcpy(d_images_K, h_images_K, sizeof(float) * n*9,
                              cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_images_R, h_images_R, sizeof(float) * n*9,
                              cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_images_C, h_images_C, sizeof(float) * n*3,
                              cudaMemcpyHostToDevice));
	
	float h_scaleRanges[12] = {1.f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f, 0.00390625f, 0.001953125f, 0.0009765625f, 0.00048828125f};
	
	checkCudaErrors(cudaMemcpyToSymbol(scaleRanges, h_scaleRanges,
										sizeof(float) * 12, 0,
										cudaMemcpyHostToDevice));
										
	h_ref_K=new float[4*n];
	h_ref_inv_K=new float[4*n];
	for (int i=0;i<n;i++){
		h_ref_K[i*4+0]=h_images_K[i*9+0];
		h_ref_K[i*4+1]=h_images_K[i*9+2];
		h_ref_K[i*4+2]=h_images_K[i*9+4];
		h_ref_K[i*4+3]=h_images_K[i*9+5];
		
		h_ref_inv_K[i*4+0]=1.0/h_ref_K[i*4+0];
		h_ref_inv_K[i*4+1]=-h_ref_K[i*4+1]*h_ref_inv_K[i*4+0];
		h_ref_inv_K[i*4+2]=1.0/h_ref_K[i*4+2];
		h_ref_inv_K[i*4+3]=-h_ref_K[i*4+3]*h_ref_inv_K[i*4+2];
	
	}
	
	//checkCudaErrors(cudaMemcpy(ref_K, h_ref_K, sizeof(float) * n*4,
    //                          cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(ref_inv_K, h_ref_inv_K, sizeof(float) * n*4,
                              cudaMemcpyHostToDevice));	
							  
	printf("SendImageData is completed\n");
}
/*
void EstimateDepthMaps_gpu::SendRefData(
		int idxImage,
		float * h_depthmap,
		float * h_confmap,
		float * h_normalmap
		
		){
			
	
	checkCudaErrors(cudaMemcpy(d_depthmap+idxImage*height*width, h_depthmap, sizeof(float) * height*width,
                              cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_confmap+idxImage*height*width, h_confmap, sizeof(float) * height*width,
                              cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_normalmap+idxImage*height*width*3, h_normalmap, sizeof(float) * height*width*3,
                              cudaMemcpyHostToDevice));
	
	thrust::copy_n(h_depthmap.begin(), height*width , depthmap.begin()+idxImage*height*width);
	thrust::copy_n(h_confmap.begin(), height*width , confmap.begin()+idxImage*height*width);
	thrust::copy_n(h_normalmap.begin(), height*width*3 , normalmap.begin()+idxImage*height*width*3);
	//printf("\n%d SendRefData is completed\n",idxImage);
}
*/
void EstimateDepthMaps_gpu::EstimateDepthMaps(
											int ref,
											float dMin,
											float dMax,
											float * h_depthmap,
											float * h_confmap,
											float * h_normalmap,
											int * h_neighbors
											){
	float * d_depthmap;
	float * d_confmap;
	float * d_normalmap;
	int * d_neighbors;
	
	checkCudaErrors(cudaMalloc(&d_confmap,   sizeof(float) * height*width));
	checkCudaErrors(cudaMalloc(&d_depthmap,   sizeof(float) * height*width));
	checkCudaErrors(cudaMalloc(&d_normalmap,   sizeof(float) * 3*height*width));
	checkCudaErrors(cudaMalloc(&d_neighbors,   sizeof(int) * h_neighbors[0]));
	
	checkCudaErrors(cudaMemcpy(d_depthmap, h_depthmap, sizeof(float) * height*width,
                              cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_confmap, h_confmap, sizeof(float) * height*width,
                              cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_normalmap, h_normalmap, sizeof(float) * height*width*3,
                              cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_neighbors, h_neighbors, sizeof(int) * h_neighbors[0],
                              cudaMemcpyHostToDevice));
	
/*	
	checkCudaErrors(cudaMemcpyToSymbol(ref_K, h_ref_K+ref*4, 
										sizeof(float) * 4, 0,
										cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol(ref_inv_K, h_ref_inv_K+ref*4,
										sizeof(float) * 4, 0,
										cudaMemcpyHostToDevice));
*/

	EstimateDepthMapsKernel::initialDepthAndNormalMap<<<ColParallelGridSize,ColParallelBlockSize>>>(
		*this,
		ref,
		dMin,
		dMax,
		d_depthmap,
		d_confmap,
		d_normalmap
	);
	synchronCheck
	
	EstimateDepthMapsKernel::initialConfMap<<<PixelParallelGridSize,PixelParallelBlockSize>>>(
		*this,
		ref,
		dMin,
		dMax,
		d_depthmap,
		d_confmap,
		d_normalmap,
		d_neighbors
	);
	
	synchronCheck
	for (int dir=0;dir<0;dir++){
		EstimateDepthMapsKernel::PropagationKernel<<<ColParallelGridSize,ColParallelBlockSize>>>(
		*this,//EstimateDepthMaps_gpu data,
		ref,//int ref,//当前的相机编号
		dir,//int dir,//0：正向，1：从后往前
		dMax,//float dMax,
		dMin,//float dMin
		d_depthmap,
		d_confmap,
		d_normalmap,
		d_neighbors
		);
		synchronCheck
	}
	
	checkCudaErrors(cudaMemcpy(h_depthmap, d_depthmap, sizeof(float) * height*width, 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_confmap, d_confmap, sizeof(float) * height*width, 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_normalmap, d_normalmap, sizeof(float) * height*width*3, 
				cudaMemcpyDeviceToHost));
				
	checkCudaErrors(cudaFree(d_confmap));
	checkCudaErrors(cudaFree(d_depthmap));
	checkCudaErrors(cudaFree(d_normalmap));
	checkCudaErrors(cudaFree(d_neighbors));
	//printf("%d EstimateDepthMaps is completed\n",ref);
}

/*
void EstimateDepthMaps_gpu::CarryOutData(
		 int idxImage,
		 //float * h_depthmap,
		 //float * h_confmap,
		 //float * h_normalmap
		thrust :: host_vector <float > &h_depthmap,
		thrust :: host_vector <float > &h_confmap,
		thrust :: host_vector <float > &h_normalmap
		 ){
	
	checkCudaErrors(cudaMemcpy(h_depthmap, d_depthmap+idxImage*height*width, sizeof(float) * height*width, 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_confmap, d_confmap+idxImage*height*width, sizeof(float) * height*width, 
				cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_normalmap, d_normalmap+idxImage*height*width*3, sizeof(float) * height*width*3, 
				cudaMemcpyDeviceToHost));
	
	thrust::copy_n(depthmap.begin()+idxImage*height*width, height*width, h_depthmap.begin());
	thrust::copy_n(confmap.begin()+idxImage*height*width, height*width, h_confmap.begin());
	thrust::copy_n(normalmap.begin()+idxImage*height*width*3, height*width*3, h_normalmap.begin());
	//printf("\n%d CarryOutData is completed\n",idxImage);
}
*/
void EstimateDepthMaps_gpu::ComputeCudaConfig(int n,int height_,int width_){
	nSizeHalfWindow=3;
	nSizeWindow=nSizeHalfWindow*2+1;
	
	ColParallelBlockSize.x = PropagationBlockCol;
  	ColParallelBlockSize.y = 1;
  	ColParallelBlockSize.z = 1;
  	ColParallelGridSize.x = (width_ - 1) / PropagationBlockCol + 1;
  	ColParallelGridSize.y = 1;
  	ColParallelGridSize.z = 1;
	
	PixelParallelBlockSize.x=32;
	PixelParallelBlockSize.y=16;
	PixelParallelBlockSize.z=1;
	PixelParallelGridSize.x=(height_-1)/32+1;
	PixelParallelGridSize.y=(width_-1)/16+1;
	PixelParallelGridSize.z=1;  
	
	randstatesBlockSize.x=16;
	randstatesBlockSize.y=32;
	randstatesBlockSize.z=1;
	randstatesGridSize.x=(n-1)/16+1;
	randstatesGridSize.y=(width-1)/32+1;
	randstatesGridSize.z=1;
}
void EstimateDepthMaps_gpu::MemoryFree(){
	//checkCudaErrors(cudaFree(d_confmap));
	//checkCudaErrors(cudaFree(d_depthmap));
	checkCudaErrors(cudaFree(d_images));
	//checkCudaErrors(cudaFree(d_normalmap));
	checkCudaErrors(cudaFree(d_images_R));
	checkCudaErrors(cudaFree(d_images_K));
	checkCudaErrors(cudaFree(d_images_C));
	free(h_ref_K);
	free(h_ref_inv_K);
	//checkCudaErrors(cudaFree(ref_K));
	checkCudaErrors(cudaFree(ref_inv_K));
	checkCudaErrors(cudaFree(randstates));
	//depthmap.clear();
	//confmap.clear();
	//normalmap.clear();
	printf("\nMemoryFree is completed\n");
}