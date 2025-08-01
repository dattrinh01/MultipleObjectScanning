#ifndef _MULTIPLE_OBJECT_SCANNING
#define _MULTIPLE_OBJECT_SCANNING

#include <iostream>
#include <vector>
#include <string>
#include <string.h>
#include <fstream>
#include <regex>
#include <algorithm>

#include <Eigen/Eigen>
#include "yaml-cpp/yaml.h"

#include <Boost/foreach.hpp>
#include <Boost/property_tree/ptree.hpp>
#include <Boost/property_tree/json_parser.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/visualization/cloud_viewer.h>


#include <pcl/features/normal_3d.h>
#include <pcl/features/cvfh.h>



/*--------------SUPPORT FUNCTIONS-----------------------*/
inline bool naturalSorting(const std::string& a, const std::string& b)
{
	if (a.empty())
		return true;
	if (b.empty())
		return false;
	if (std::isdigit(a[0]) && !std::isdigit(b[0]))
		return true;
	if (!std::isdigit(a[0]) && std::isdigit(b[0]))
		return false;
	if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
	{
		if (std::toupper(a[0]) == std::toupper(b[0]))
			return naturalSorting(a.substr(1), b.substr(1));
		return (std::toupper(a[0]) < std::toupper(b[0]));
	}

	/*Both strings begin with digit --> parse both numbers*/
	std::istringstream issa(a);
	std::istringstream issb(b);
	int ia, ib;
	issa >> ia;
	issb >> ib;
	if (ia != ib)
		return ia < ib;

	/*Numbers are the same --> remove numbers and recurse*/
	std::string anew, bnew;
	std::getline(issa, anew);
	std::getline(issb, bnew);
	return (naturalSorting(anew, bnew));
}
void eraseSubStrings(std::string& mainString, std::string& toErase);
pcl::PointCloud<pcl::PointXYZ>::Ptr createPointCloud(cv::Mat depth_img, const double depth_intrinsic[4]);
bool checkSubString(std::string mainString, std::string checkString);
void extractBoundingBoxFromMaskImage(cv::Mat mask_img, double& bbX, double& bbY, double& bbWidth, double& bbHeight);
pcl::PointCloud<pcl::PointXYZ>::Ptr generatePointCloudFromDepthImage(cv::Mat depth_img, const double depth_intrinsic[4]);
void cropTheObjectFromDepthImage(cv::Mat& depth_img, double boundingBox[4]);

pcl::PointCloud<pcl::PointXYZ> convertEigenMatrixXdToPCLCloud(const Eigen::MatrixXd& inputMatrix);
Eigen::MatrixXd convertPCLCloudToEigenMatrixXd(const pcl::PointCloud<pcl::PointXYZ>::Ptr& inputCloud);
Eigen::MatrixXd transformPointsWithTransformMatrix(const Eigen::MatrixXd& inputVertices, const Eigen::Matrix4f& transformMatrix);
Eigen::Matrix4f transformVerticesFromPointToPoint(const Eigen::MatrixXd& targetVertices, const Eigen::Vector3d fromPoint, const Eigen::Vector3d toPoint, Eigen::MatrixXd& outPoints);
void transformPointCloudToOriginal(pcl::PointCloud<pcl::PointXYZ>::Ptr& inCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& outCloud);
void readCameraMatrix(std::string infoYMLFilePath, std::vector<float>& infoMatrix, std::string obj_name);
void configBoundingBox(std::string gtFilePath, std::vector<double>& boundingBoxVec, std::string obj_name, int className);
void removeLeadingZeros(std::string& str);
/*--------------MAIN PROCESSING FUNCTION----------------*/
void datasetGeneration();
void detectMultipleObjects();
void cutPointCloudOfDetectObjects();
void mergePointClouds();
void meshPointClouds();
void evaluationPointClouds();

void test_func();
/*--------------MAIN FUNCTIONS--------------------------*/
void mainFunction();

#endif // !_MULTIPLE_OBJECT_SCANNING
