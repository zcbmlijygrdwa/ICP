
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
#include "header.h"  

#include <stdio.h>
#include <float.h>

#include "../../include/shared/test.hpp"


void cudaFindNearest(int numBlocks, int threadsPerBlock, double *P, double *Q, int nP, int nQ, double *Q_select, int *min_index_device);
__global__ void kernelIterativeClosestPoint(double *P, double *Q, int nP, int nQ, int pointsPerThread, double *Q_select_device,int *min_index_device);
Eigen::Matrix4d GetTransform(double *Pselect, double *Qselect, int);
void Transform(double *P, const Eigen::MatrixXd Transmatrix, int , double *);


// Catch the cuda error
#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s at %s:%d\n",
			cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#else
#define cudaCheckError(ans) ans
#endif


/******************************************************
º¯Êý¹¦ÄÜ£ºICPËã·¨£¬¼ÆËãÁ½¸öµãÔÆÖ®¼äµÄ×ª»»¹ØÏµ
ÊäÈë²ÎÊý£ºcloud_targetÄ¿±êµãÔÆ£¬cloud_sourceÔ­Ê¼µãÔÆ
Iterµü´ú²ÎÊý
Êä³ö²ÎÊý£ºtransformation_matrix ×ª»»²ÎÊý
********************************************************/
void feezhu_icp(Eigen::MatrixXd cloud_target,
	Eigen::MatrixXd cloud_source,
	const Iter_para Iter, Eigen::Matrix4d &transformation_matrix)
{

	//1.Ñ°ÕÒPÖÐµãÔÚQÖÐ¾àÀë×î½üµÄµã
	int nP = cloud_target.cols();
	int nQ = cloud_source.cols();
	int p_size = sizeof(double) * nP * 3;//p size
	int q_size = sizeof(double) * nQ * 3;
    //printv(nP);
    //printv(nQ);

	/*Data on host*/
	Eigen::MatrixXd P (cloud_target);
	double *P_origin = P.data();

	double *P_host = cloud_target.data();
	double *Q_host = cloud_source.data();

	double * Q_select = (double *)malloc(p_size);

	/*Data on device*/
	double * P_device;
	double * Q_device;
	double * Q_selectdevice;
	int* min_index_device;

	/*Malloc space in gpu*/
	cudaMalloc(&P_device, p_size);
	cudaMalloc(&Q_device, q_size);
	cudaMalloc(&Q_selectdevice, p_size);
	cudaMalloc(&min_index_device, sizeof(int) * nP);

	/*copy data from memory to cuda*/
	cudaMemcpy(Q_device, Q_host, q_size, cudaMemcpyHostToDevice);

	/* set cuda block*/
	int numBlocks = 32;
	int threadsPerBlock =64;
	//int numBlocks = 128;
	//int threadsPerBlock = 1024;

	int i = 1;
	while (i < Iter.Maxiterate)
	{
		//printf("iteration : %d\n", i);
		//gpu
		/*copy selectP data from memory to cuda*/
		cudaMemcpy(P_device, P_host, p_size, cudaMemcpyHostToDevice);
		/* Find cloest poiny in cloudsource*/
        //printv("here is time for find corr");
        //tic();
		cudaFindNearest(numBlocks, threadsPerBlock, P_device, Q_device, nP, nQ, Q_selectdevice, min_index_device);
        //toc();
		/* copy the Q_select*/
		cudaError_t status = cudaMemcpy(Q_select, Q_selectdevice, p_size, cudaMemcpyDeviceToHost);
		if (status == cudaSuccess) { printf("ÓÐÐ§"); }
		//cpu
		//2.Çó½â¶ÔÓ¦µÄ¸ÕÌå±ä»»
        //printv("here is time for get transform");
        //tic();
		transformation_matrix = GetTransform(P_host, Q_select, nP);
        //toc();
		//3.¶ÔP×ö±ä»»µÃµ½ÐÂµÄµãÔÆ
		Transform(P_host, transformation_matrix, nP, P_host);

		////3.¸ÕÌå±ä»»µÄ²¢ÐÐÊµÏÖ
		//double *transformation_matrix_host = transformation_matrix.data();
		//cudaMemcpy(P_device, P_host, p_size, cudaMemcpyHostToDevice);
		//cuTransform(numBlocks, threadsPerBlock, P_device, transformation_matrix, nP);

		//4.µü´úÉÏÊö¹ý³ÌÖ±ÖÁÊÕÁ²
        double error = abs(transformation_matrix(0,3))
        + abs(transformation_matrix(1,3))
        + abs(transformation_matrix(2,3))
        + abs(transformation_matrix(0,0) - 1)
        + abs(transformation_matrix(1,1) - 1)
        + abs(transformation_matrix(2,2) - 1);
        //printv(error);
		//if (abs(error) < Iter.ControlN*Iter.acceptrate*Iter.threshold)//80%µãÎó²îÐ¡ÓÚ0.01
		if (abs(error) < 0.00001)
		{
			break;
		}
		i++;
	}
	transformation_matrix = GetTransform(P_origin, P_host,nP);
	cudaFree(P_device);
	cudaFree(Q_device);
	cudaFree(Q_selectdevice); 
	cudaFree(min_index_device);
}

/******************************************************
º¯ÊýÃèÊö£º¼ÆËãÁ½¸öµãÔÆÖ®¼ä×î½üµãµÄ¾àÀëÎó²î,GPUºËº¯Êý
ÊäÈë²ÎÊý£ºcloud_targetÄ¿±êµãÔÆ¾ØÕó£¬cloud_sourceÔ­Ê¼µãÔÆ¾ØÕó
Êä³ö²ÎÊý£ºerror ×î½üµã¾àÀëÎó²îºÍÎó²îº¯ÊýµÄÖµ,ConQÓëP¶ÔÓ¦µÄ¿ØÖÆµã¾ØÕó
********************************************************/
void cudaFindNearest(int numBlocks, int threadsPerBlock, double *P, double *Q, int nP, int nQ, double *Q_select, int *min_index_device) {
	/* Assign points to each thread */
	int pointsPerThread = (nP + numBlocks * threadsPerBlock - 1) / (numBlocks * threadsPerBlock);

	//printf("%d\n", pointsPerThread);
	kernelIterativeClosestPoint << <numBlocks, threadsPerBlock >> > (P, Q, nP, nQ, pointsPerThread, Q_select, min_index_device);
	cudaCheckError(cudaThreadSynchronize());

}

__global__ void kernelIterativeClosestPoint(double *P, double *Q, int nP, int nQ, int pointsPerThread, double *Q_select_device, int *min_index_device)
{

	//__shared__ int min_index_device[N];
	//__syncthreads();
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < pointsPerThread; i++) {
		/* Handle exceptions */
		int pIdx = idx * pointsPerThread + i; // The location in P
		if (pIdx < nP) {
			/* For each point in Q */
			double minDist = FLT_MAX; // Change this later
			int minIndex = -1;
			int pValIdx = pIdx * 3;
			for (int j = 0; j < nQ; j++) {
				int qValIdx = j * 3;
				double dx = P[pValIdx] - Q[qValIdx];
				double dy = P[pValIdx + 1] - Q[qValIdx + 1];
				double dz = P[pValIdx + 2] - Q[qValIdx + 2];
				double dist = sqrtf(dx*dx + dy*dy + dz*dz);
				/* Update the nearest point */
				if (dist < minDist) {
					minDist = dist;
					minIndex = j;
				}
			}
			min_index_device[pIdx] = minIndex;
		}
	}

	//__syncthreads(); 
	/* Copy the data to Qselect */
	for (int i = 0; i < pointsPerThread; i++) {
		int pIdx = idx * pointsPerThread + i;
		if (pIdx < nP) {
			int qIdx = min_index_device[pIdx];
			int qValIdx = qIdx * 3;
			Q_select_device[pIdx * 3] = Q[qValIdx];
			Q_select_device[pIdx * 3 + 1] = Q[qValIdx + 1];
			Q_select_device[pIdx * 3 + 2] = Q[qValIdx + 2];
		}
	}
}


/******************************************************
º¯ÊýÃèÊö£ºÇóÁ½¸öµãÔÆÖ®¼äµÄ±ä»»¾ØÕó
ÊäÈë²ÎÊý£ºConPÄ¿±êµãÔÆ¿ØÖÆµã3*N£¬ConQÔ­Ê¼µãÔÆ¿ØÖÆµã3*N
Êä³ö²ÎÊý£ºtransformation_matrixµãÔÆÖ®¼ä±ä»»²ÎÊý4*4
********************************************************/
Eigen::Matrix4d GetTransform(double *Pselect, double *Qselect, int nsize)
{

	Eigen::MatrixXd ConP = Map<Eigen::MatrixXd>(Pselect, 3, nsize);
	Eigen::MatrixXd ConQ = Map<Eigen::MatrixXd>(Qselect, 3, nsize);
	//ÇóµãÔÆÖÐÐÄ²¢ÒÆµ½ÖÐÐÄµã
	Eigen::VectorXd MeanP = ConP.rowwise().mean();
	Eigen::VectorXd MeanQ = ConQ.rowwise().mean();
	//cout << MeanP <<endl<< MeanQ << endl;
	Eigen::MatrixXd ReP = ConP.colwise() - MeanP;
	Eigen::MatrixXd ReQ = ConQ.colwise() - MeanQ;
	//Çó½âÐý×ª¾ØÕó
	//Eigen::MatrixXd H = ReQ*(ReP.transpose());
	Eigen::MatrixXd H = ReP*(ReQ.transpose());
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Matrix3d V = svd.matrixV();
	double det = (U * V.transpose()).determinant();
	Eigen::Vector3d diagVec(1.0, 1.0, det);
	Eigen::MatrixXd R = V * diagVec.asDiagonal() * U.transpose();
	//Eigen::MatrixXd R = H*((ReP*(ReP.transpose())).inverse());
	//Çó½âÆ½ÒÆÏòÁ¿
	Eigen::VectorXd T = MeanQ - R*MeanP;

	Eigen::MatrixXd Transmatrix = Eigen::Matrix4d::Identity();
	Transmatrix.block(0, 0, 3, 3) = R;
	Transmatrix.block(0, 3, 3, 1) = T;
	///cout << Transmatrix << endl;
	return Transmatrix;
}



/******************************************************
º¯ÊýÃèÊö£ºµãÔÆ±ä»»
ÊäÈë²ÎÊý£ºConPµãÔÆ¿ØÖÆµã3*N£¬transformation_matrixµãÔÆÖ®¼ä±ä»»²ÎÊý4*4
Êä³ö²ÎÊý£ºNewPÐÂµÄµãÔÆ¿ØÖÆµã3*N
********************************************************/
void Transform(double *P, const Eigen::MatrixXd Transmatrix,int nsize, double *newP)
{
	Eigen::MatrixXd R = Transmatrix.block(0, 0, 3, 3);
	Eigen::VectorXd T = Transmatrix.block(0, 3, 3, 1);

	////double *NewP= (double *)malloc(3*nsize * sizeof(double));
	for (int i = 0; i < nsize; i++)
	{
		int ValIdx = i * 3;
		newP[ValIdx] = R(0, 0)*P[ValIdx] + R(0, 1)*P[ValIdx + 1] + R(0, 2)*P[ValIdx + 2] + T[0];
		newP[ValIdx+1] = R(1, 0)*P[ValIdx] + R(1, 1)*P[ValIdx + 1] + R(1, 2)*P[ValIdx + 2] + T[1];
		newP[ValIdx+2] = R(2, 0)*P[ValIdx] + R(2, 1)*P[ValIdx + 1] + R(2, 2)*P[ValIdx + 2] + T[2];
	}
	//Eigen::MatrixXd ConP = Map<Eigen::MatrixXd>(P, 3, nsize);
	//Eigen::MatrixXd NewP = (R*ConP).colwise() + T;
	//newP = NewP.data();
}


__global__ void kernelTransform()
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
}

void Getinfo()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
    printf("¿¿¿¿¿¿cuda¿¿¿¿¿¿%d\n", count);
	for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
		printf("----¿%d¿¿¿¿¿¿¿¿¿----\n", i + 1);
		printf("processor name%s \n", prop.name);
		printf("computation power%d.%d\n", prop.major, prop.minor);
		printf("global memory:%dMB\n", prop.totalGlobalMem / 1024 / 1024);
		printf("constat memory:%dKB\n", prop.totalConstMem / 1024);
		printf("¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿%dKB\n", prop.sharedMemPerBlock / 1024);
		printf("¿¿¿¿¿¿¿¿¿¿¿¿¿%d\n", prop.warpSize);
		printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
		printf("¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿(%d,%d,%d)\n", prop.maxThreadsDim[0],
			prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿(%d,%d,%d)\n", prop.maxGridSize[0],
			prop.maxGridSize[1], prop.maxGridSize[2]);
	}
}
