//#pragma once
#include <fstream> 
#include "header.h"
#include "/home/zhenyu/catkin_ws/src/pointcloud-georeference/valk_lidar/shared/include/shared/test.hpp"

Eigen::MatrixXd Transform(const Eigen::MatrixXd ConP, const Eigen::MatrixXd Transmatrix)
{
	Eigen::MatrixXd R = Transmatrix.block(0, 0, 3, 3);
	Eigen::VectorXd T = Transmatrix.block(0, 3, 3, 1);

	Eigen::MatrixXd NewP = (R*ConP).colwise() + T;
	return NewP;
}

void print4x4Matrix(const Eigen::Matrix4d & matrix)
{
	printf("Rotation matrix :\n");
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
	printf("Translation vector :\n");
	printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

Eigen::MatrixXd ReadFile(std::string FileName)
{

	Eigen::MatrixXd cloud(3,N);
	
	std::ifstream fin(FileName);
	if (!fin.is_open())
	{
		cout << "Error:can not open the file!";
		exit(1);
	}
	int i = 0;
	//while (!fin.eof())
    for(int i = 0 ; i < N ; i++)
	{
        printv(i);
		fin >> cloud(0,i) >> cloud(1,i) >> cloud(2,i);
		//i++;
	}

    //printv(cloud.rows());
    //printv(cloud.cols());
    //printv(cloud.transpose());

	return cloud;
}

int main(int argc, char** argv)
{
	// read point cloud from txt file
    printv(1);
	//Eigen::MatrixXd cloud_in = ReadFile("/home/zhenyu/ICP/data/bunny.txt");
	Eigen::MatrixXd cloud_in = ReadFile("/home/zhenyu/ICP/data/bunny_backup.txt");
    printv(2);
	Eigen::MatrixXd cloud_icp;

	// Defining a rotation matrix and translation vector
	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
    printv(transformation_matrix);

	// A rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
	double theta = M_PI / 8;  // The angle of rotation in radians
	//double theta = 0;
	transformation_matrix(0, 0) = cos(theta);
	transformation_matrix(0, 1) = -sin(theta);
	transformation_matrix(1, 0) = sin(theta);
	transformation_matrix(1, 1) = cos(theta);

	// A translation on Z axis (0.4 meters)
	transformation_matrix(0, 3) = -0.05;
	transformation_matrix(1, 3) = 0.02;
	transformation_matrix(2, 3) = 0.01;
	// Display in terminal the transformation matrix
	std::cout << "Applying this rigid transformation to: cloud_in -> cloud_icp" << std::endl;
	print4x4Matrix(transformation_matrix);

	cloud_icp=Transform(cloud_in, transformation_matrix);
	
	//icp algorithm
	Eigen::Matrix4d matrix_icp;
	Iter_para iter{ N,30,0.00001,0.8 };
	Getinfo();
	long begin = clock();
	icp(cloud_in, cloud_icp, iter, matrix_icp);
	std::cout << "GPU time used" << int(((double)(clock() - begin)) / CLOCKS_PER_SEC * 1000) << "ms " << std::endl;
	cout << "matrix_icp = \n"<<matrix_icp << endl;

    return 0;
}
