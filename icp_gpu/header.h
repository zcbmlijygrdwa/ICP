//#pragma once
#include<time.h>
#include <iostream>  
#include <vector>
#include <Eigen/Dense>
//#define N 9
//#define N 3594
#define LINE_NUM 35947
//#define N 761
//#define M_PI 3.1415926

struct Iter_para //Interation paraments
{
	int ControlN;
	int Maxiterate;
	double threshold;
	double acceptrate;

};

using namespace std;
using Eigen::Map;

void feezhu_icp(const Eigen::MatrixXd cloud_target,
	const Eigen::MatrixXd cloud_source,
	const Iter_para Iter, Eigen::Matrix4d &transformation_matrix);
void Getinfo();


