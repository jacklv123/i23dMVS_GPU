//this class is aim to fuse depthmaps


const float eps=1e-8;

class FuseDepthMaps_gpu{
public:
    FuseDepthMaps_gpu();
    ~FuseDepthMaps_gpu();
    void hello();
    void Malloc_memory(
			int n,
			int nPointsEstimate,
			int height,
			int width
		);
    void copy_input_data(
			int * h_connection_data,
			int n,
			float * h_images_K,
			float * h_images_R,
			float * h_images_C,
			float * h_depthmap,
			float * h_confmap,
			int * h_neighbors,
			int * h_neighbors_begin,
			int * h_neighbors_end
		);
	void Carry_Out(
			float * h_points,
			float * h_weight,
			int * h_view,
			int * h_pointsinfo,
			int n,
			int nPointsEstimate
		);
	
    __device__ float3 TransformPointW2C(int ref,float3 point);
    __device__ float3 TransformPointC2I(int ref,float3 point);
    __device__ float3 TransformPointI2W(int ref,float3 point);
    __device__ float3 TransformPointI2C(int ref,float3 point);
    __device__ float3 TransformPointC2W(int ref, float3 point);
    __device__ int IsDepthSimilar(float d0,float d1);
    

    void FuseDepthMaps(int n);

    void ComputeCudaConfig(int height_,int width_);
    void MemoryFree(int n,int nPointsEstimate);
//protected:
    
    const static size_t kBlockDimX = 32;
    const static size_t kBlockDimY = 16;

    dim3 blockSize_;
    dim3 gridSize_;

    int * connection_data;
/////输出结果

    float * d_points;
    int * d_view;
    float * d_weight;
	int *d_pointsinfo;
    //0:writepos of points,=3*number_of_points
	//1:writepos of view and weight
	//2-..每个点的view的数量
	//每段view数据的第一个数据是对应的point的idx
///////////这些数据来自scene.images,通过图片的idx来找到相机参数
    float * d_images_K;//3×3的矩阵
    float * d_images_R;//3×3的矩阵
    float * d_images_C;//1×3的向量


/////////////////////////////////////////////////////////
    //int ** size;//记录遍历图片的height和width
       //先假设大小均一样

///////////来自arrDepthData[idxImage]，是深度图数据，?ü计卤暾业缴疃韧?
    float * d_depthmap;

//////////来自arrDepthData[idxImage],是置信图，通过图片下标找到置信图
    float * d_confmap;

/////////来自arrDepthData[idxImage],是相邻图片集
    int * d_neighbors;
//neighbors的在数组中的开始和结束位置[start,end)
        int * d_neighbors_begin;
	int * d_neighbors_end;
////////////////////////////////////////////////////////////////////////////////长宽
	int height,width;
};

namespace kernel {

__global__ void FuseDepthMaps_kernel(
	FuseDepthMaps_gpu depthdata,
	int ref,//当前的相机编号
	int height,
	int width,
	int n,
	float * weight,
	int * view
)
{
	
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
  	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("#################\n");
	const  int pos =ref*height*width+row*width+col;
	const int image_pos=row*width+col;

	//当前的像素点坐标
	//在图片的边界内
	if (row>=0&&row < height &&col>=0&& col < width ) {
	} else {
		return ;
	}

	float depth=depthdata.d_depthmap[pos];
	if ( depth==0 ) return ;

	float conf=depthdata.d_confmap[pos];

	int view_size=0;	
	float3 ans;
	float3 world;
	float3 X;
	float3 tmp;

	tmp.x=col;
	tmp.y=row;
	tmp.z=depth;

	world=depthdata.TransformPointI2W(
		ref,
		tmp
	);

	
	view[image_pos*n+view_size]=ref;
	weight[image_pos*n+view_size]=conf;
	
	view_size++;

	X.x=conf*world.x;
	X.y=conf*world.y;
	X.z=conf*world.z;
		
	for (int i=depthdata.d_neighbors_begin[ref];i<depthdata.d_neighbors_end[ref];i++)
	{
		int idxB=depthdata.d_neighbors[i];
		tmp=depthdata.TransformPointW2C(
			idxB,
			world
		);

		ans=depthdata.TransformPointC2I(
			idxB,
			tmp
		);
		int h=int(ans.y+0.5);
		int w=int(ans.x+0.5);
		
		if (h>=0&&h<height&&w>=0&&w<width){} else continue;
		float depthB=depthdata.d_depthmap[idxB*height*width+h*width+w];
		if (depthB==0) continue;
		float depth=tmp.z;
		if (depthdata.IsDepthSimilar(depth,depthB))
		{
			float confB=depthdata.d_confmap[idxB*height*width+h*width+w];
			view[image_pos*n+view_size]=idxB;
			weight[image_pos*n+view_size]=confB;
			view_size++;

			tmp=ans;
			tmp.z=depthB;

			ans=depthdata.TransformPointI2W(
				idxB,
				tmp
			);
			X.x=X.x+ans.x*confB;
			X.y=X.y+ans.y*confB;
			X.z=X.z+ans.z*confB;
			conf=conf+confB;
			depthdata.d_depthmap[idxB*height*width+h*width+w]=0;
		} else 
		if (depth<depthB)
		{
			//printf("%lf %lf\n",depth,depthB);
			//遮挡
			depthdata.d_depthmap[idxB*height*width+h*width+w]=0;
		}	
	}
	
	X.x=X.x/conf;
	X.y=X.y/conf;
	X.z=X.z/conf;
	
	if (view_size<2)
	{
		//printf("%d\n",view_size);
		//不添加这个点
	} else {
		int pos=atomicAdd(&depthdata.d_pointsinfo[0],3);
		depthdata.d_points[pos]=X.x;
		depthdata.d_points[pos+1]=X.y;
		depthdata.d_points[pos+2]=X.z;
		int idx=pos/3+1;

		depthdata.d_pointsinfo[idx*2]=view_size;

		pos=atomicAdd(&depthdata.d_pointsinfo[1],view_size);
		
		depthdata.d_pointsinfo[idx*2+1]=pos;
				
		for (int i=0;i<view_size;i++)
			depthdata.d_weight[pos+i]=weight[image_pos*n+i];
			
		for (int i=0;i<view_size;i++)
			depthdata.d_view[pos+i]=view[image_pos*n+i];
				
	} 
}

}//namespace


FuseDepthMaps_gpu::FuseDepthMaps_gpu(){
}
FuseDepthMaps_gpu::~FuseDepthMaps_gpu(){

}
void FuseDepthMaps_gpu::ComputeCudaConfig(int height_,int width_) {
  blockSize_.x = kBlockDimX;
  blockSize_.y = kBlockDimY;
  blockSize_.z = 1;

  gridSize_.x = (width_ - 1) / kBlockDimX + 1;
  gridSize_.y = (height_ - 1) / kBlockDimY + 1;
  gridSize_.z = 1;
}
void FuseDepthMaps_gpu::Malloc_memory(int n,int nPointsEstimate,int Height,int Width){
	height=Height;
	width=Width;

	connection_data=new int[n];

	checkCudaErrors(cudaMalloc(&d_neighbors_begin,   sizeof(int) * n));
	checkCudaErrors(cudaMalloc(&d_neighbors_end,   sizeof(int) * n));
	checkCudaErrors(cudaMalloc(&d_neighbors,   sizeof(int) * n*n));

	

	checkCudaErrors(cudaMalloc(&d_images_K,   sizeof(float)*9*n));
	checkCudaErrors(cudaMalloc(&d_images_R,   sizeof(float)*9*n));
	checkCudaErrors(cudaMalloc(&d_images_C,   sizeof(float)*3*n));

	checkCudaErrors(cudaMalloc(&d_depthmap,   sizeof(float)*height*width*n));
	checkCudaErrors(cudaMalloc(&d_confmap,   sizeof(float)*height*width*n));

	checkCudaErrors(cudaMalloc(&d_points,   sizeof(float)*nPointsEstimate*3));
	checkCudaErrors(cudaMalloc(&d_weight,   sizeof(float)*nPointsEstimate*(n+1)));
	checkCudaErrors(cudaMalloc(&d_view,   sizeof(int)*nPointsEstimate*(n+1)));
	checkCudaErrors(cudaMalloc(&d_pointsinfo,   sizeof(int)*nPointsEstimate*2));
	
	printf("Malloc_memory is completed\n");
}
void FuseDepthMaps_gpu::copy_input_data(
			int * h_connection_data,
			int n,
			float * h_images_K,
			float * h_images_R,
			float * h_images_C,
			float * h_depthmap,
			float * h_confmap,
			int * h_neighbors,
			int * h_neighbors_begin,
			int * h_neighbors_end
)
{	
	memcpy(connection_data,h_connection_data,sizeof(int)*n);

	//neighbors data
	checkCudaErrors(cudaMemcpy(d_neighbors, h_neighbors, sizeof(int) * n*n,
                              cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMemcpy(d_neighbors_begin, h_neighbors_begin, sizeof(int) * n,
                              cudaMemcpyHostToDevice));		
			  
	checkCudaErrors(cudaMemcpy(d_neighbors_end, h_neighbors_end, sizeof(int) * n,
                              cudaMemcpyHostToDevice));	

	//camera data
	checkCudaErrors(cudaMemcpy(d_images_K, h_images_K, sizeof(int) * 9*n,
                              cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_images_R, h_images_R, sizeof(int) * 9*n,
                              cudaMemcpyHostToDevice));
		
	checkCudaErrors(cudaMemcpy(d_images_C, h_images_C, sizeof(int) * 3*n,
                              cudaMemcpyHostToDevice));

	//depthdata & confmap
	checkCudaErrors(cudaMemcpy(d_depthmap, h_depthmap, sizeof(float)*height*width*n,
                              cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_confmap, h_confmap, sizeof(float)*height*width*n,
                              cudaMemcpyHostToDevice));
	
	printf("copy_input_data is completed\n");
}
void FuseDepthMaps_gpu::Carry_Out(
			float * h_points,
			float * h_weight,
			int * h_view,
			int * h_pointsinfo,
			int n,
			int nPointsEstimate
)
{
	checkCudaErrors(cudaMemcpy(h_points, d_points, sizeof(float)*nPointsEstimate*3, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_weight, d_weight, sizeof(float)*nPointsEstimate*(n+1), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_view, d_view, sizeof(int)*nPointsEstimate*(n+1), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_pointsinfo, d_pointsinfo, sizeof(int)*nPointsEstimate*2, cudaMemcpyDeviceToHost));
	printf("Carry_Out is completed!\n");
}
__device__ float3 FuseDepthMaps_gpu::TransformPointW2C(int ref,float3  point)
{	
	float3 ans;
	ans.x=d_images_R[ref*9+0]*(point.x-d_images_C[ref*3+0])+d_images_R[ref*9+1]*(point.y-d_images_C[ref*3+1])+d_images_R[ref*9+2]*(point.z-d_images_C[ref*3+2]);
	ans.y=d_images_R[ref*9+3]*(point.x-d_images_C[ref*3+0])+d_images_R[ref*9+4]*(point.y-d_images_C[ref*3+1])+d_images_R[ref*9+5]*(point.z-d_images_C[ref*3+2]);
	ans.z=d_images_R[ref*9+6]*(point.x-d_images_C[ref*3+0])+d_images_R[ref*9+7]*(point.y-d_images_C[ref*3+1])+d_images_R[ref*9+8]*(point.z-d_images_C[ref*3+2]);
	return ans;
}
__device__ float3 FuseDepthMaps_gpu::TransformPointC2I(int ref,float3 point)
{
	float3 ans;
	ans.x=d_images_K[ref*9+2]+d_images_K[ref*9+0]*(point.x/point.z);
	ans.y=d_images_K[ref*9+5]+d_images_K[ref*9+4]*(point.y/point.z);
	return ans;
}
__device__ float3 FuseDepthMaps_gpu::TransformPointI2W(int ref,float3 point)
{
	float3 a;
	a=TransformPointI2C(ref,point);
	a=TransformPointC2W(ref,a);
	return a;
}
__device__ float3 FuseDepthMaps_gpu::TransformPointI2C(int ref,float3 point)
{
	float3 ans;
	ans.x=(point.x-d_images_K[ref*9+2])*point.z/d_images_K[ref*9+0];
	ans.y=(point.y-d_images_K[ref*9+5])*point.z/d_images_K[ref*9+4];
	ans.z=point.z;
	return ans;
}
__device__ float3 FuseDepthMaps_gpu::TransformPointC2W(int ref,float3 point)
{
	float3 ans;
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

	float A0=(b2*c3-c2*b3)*co;
	float A1=(c1*b3-b1*c3)*co;
	float A2=(b1*c2-c1*b2)*co;
	float A3=(c2*a3-a2*c3)*co;
	float A4=(a1*c3-c1*a3)*co;
	float A5=(a2*c1-a1*c2)*co;
	float A6=(a2*b3-b2*a3)*co;
	float A7=(b1*a3-a1*b3)*co;
	float A8=(a1*b2-a2*b1)*co;
	ans.x=A0*point.x+A1*point.y+A2*point.z+d_images_C[ref*3+0];
	ans.y=A3*point.x+A4*point.y+A5*point.z+d_images_C[ref*3+1];
	ans.z=A6*point.x+A7*point.y+A8*point.z+d_images_C[ref*3+2];
	return ans;
}
__device__ int FuseDepthMaps_gpu::IsDepthSimilar(float d0,float d1)
{
	float a=d0-d1;
	if (a<0) a=-a;
	a=a/d0;
	//printf("%lf\n",a);
	if (a<0.01) return 1;
	return 0;
}	
void FuseDepthMaps_gpu::MemoryFree(int n,int nPointsEstimate)
{
	checkCudaErrors(cudaFree(d_neighbors));
	checkCudaErrors(cudaFree(d_neighbors_begin));
	checkCudaErrors(cudaFree(d_neighbors_end));

	checkCudaErrors(cudaFree(d_images_K));
	checkCudaErrors(cudaFree(d_images_R));
	checkCudaErrors(cudaFree(d_images_C));

	checkCudaErrors(cudaFree(d_depthmap));
	checkCudaErrors(cudaFree(d_confmap));

	checkCudaErrors(cudaFree(d_points));
	checkCudaErrors(cudaFree(d_weight));
	checkCudaErrors(cudaFree(d_view));
	checkCudaErrors(cudaFree(d_pointsinfo));

	free(connection_data);
	printf("memory is free\n");
}

void FuseDepthMaps_gpu::FuseDepthMaps(int n)
{
	float *weight=NULL;
	int *view=NULL;

	checkCudaErrors(cudaMalloc(&weight,   sizeof(float)*height*width*n));
	checkCudaErrors(cudaMalloc(&view,   sizeof(int)*height*width*n));
	for (int i=0;i<n;i++)
	{
		//printf("calculate the ith image : %d   |--------------\n",i);
		int ref=connection_data[i];
		//printf("%d %d %d",ref,height,width);
		ComputeCudaConfig(height,width);//设置好gridsize和blocksize
		//printf("gridSize is :(%d %d %d)    ###    blockSize is :(%d %d %d)\n", gridSize_.x,gridSize_.y,gridSize_.z,blockSize_.x,blockSize_.y,blockSize_.z);
		kernel::FuseDepthMaps_kernel<<<gridSize_,blockSize_>>>(*this,ref,height,width,n,weight,view);

	}
	checkCudaErrors(cudaFree(weight));
	checkCudaErrors(cudaFree(view));

	printf("FuseDepthMaps is completed\n");
}

void FuseDepthMaps_gpu::hello()
{
	//kernel::FuseDepthMaps_kernel<<<1,15>>>();
}