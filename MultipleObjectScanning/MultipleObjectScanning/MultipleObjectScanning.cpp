#include "MultipleObjectScanning.h"

/*--------------SUPPORT FUNCTIONS-----------------------*/
void eraseSubStrings(std::string& mainString, std::string& toErase)
{
	std::size_t position = mainString.find(toErase);
	if (position != std::string::npos)
	{
		mainString.erase(position, toErase.length());
	}
}

bool checkSubString(std::string mainString, std::string checkString)
{
	if (strstr(mainString.c_str(), checkString.c_str()))
	{
		return true;
	}
	else
	{
		return false;
	}
}

void extractBoundingBoxFromMaskImage(cv::Mat mask_img, double& bbX, double& bbY, double& bbWidth, double& bbHeight)
{
	cv::Mat mSource_Gray, mThreshold;
	cv::cvtColor(mask_img, mSource_Gray, cv::COLOR_BGR2GRAY);
	cv::threshold(mSource_Gray, mThreshold, 254, 255, cv::THRESH_BINARY_INV);
	cv::Mat Points;
	findNonZero(mThreshold, Points);
	cv::Rect Min_Rect = boundingRect(Points);

	double x_min, x_max, y_min, y_max;
	x_min = Min_Rect.tl().x;
	y_min = Min_Rect.tl().y;
	x_max = Min_Rect.br().x;
	y_max = Min_Rect.br().y;

	bbX = (x_min + x_max) / 2.0 / mask_img.size().width;
	bbY = (y_min + y_max) / 2.0 / mask_img.size().height;
	bbWidth = (x_max - x_min) / mask_img.size().width;
	bbHeight = (y_max - y_min) / mask_img.size().height;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr generatePointCloudFromDepthImage(cv::Mat depth_img, double depth_intrinsic[4])
{
	double fx = depth_intrinsic[0];
	double fy = depth_intrinsic[1];
	double cx = depth_intrinsic[2];
	double cy = depth_intrinsic[3];

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	for (int y = 0; y < depth_img.rows; y++) {
		for (int x = 0; x < depth_img.cols; x++) {
			pcl::PointXYZ p;
			ushort depth_val = depth_img.ptr<ushort>(y)[x];
			if (depth_val == 0) { continue; }
			p.z = depth_val * 0.1;
			p.x = (x - cx) * p.z / fx;
			p.y = (y - cy) * p.z / fy;
			cloud->points.push_back(p);
		}
	}
	return cloud;
}

void cropTheObjectFromDepthImage(cv::Mat& depth_img, double boundingBox[4])
{
	for (int y = 0; y < depth_img.rows; y++) {
		for (int x = 0; y < depth_img.cols; x++) {

			if (!(y >= boundingBox[1] && y <= boundingBox[3] && x >= boundingBox[0] && x <= boundingBox[2]))
			{
				depth_img.ptr<ushort>(y)[x] = 0;
			}
		}
	}
}

pcl::PointCloud<pcl::PointXYZ> convertEigenMatrixXdToPCLCloud(const Eigen::MatrixXd& inputMatrix)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud(new pcl::PointCloud<pcl::PointXYZ>());
	outputCloud->points.resize(inputMatrix.rows());

	outputCloud->height = 1;
	outputCloud->width = inputMatrix.rows();

	for (unsigned int i = 0; i < outputCloud->points.size(); i++) {
		pcl::PointXYZ point;
		point.x = inputMatrix(i, 0);
		point.y = inputMatrix(i, 1);
		point.z = inputMatrix(i, 2);
		outputCloud->points[i] = point;
	}
	return *outputCloud;
}

Eigen::MatrixXd convertPCLCloudToEigenMatrixXd(const pcl::PointCloud<pcl::PointXYZ>::Ptr& inputCloud)
{
	Eigen::MatrixXd outputBuffer(inputCloud->points.size(), 3);
	for (unsigned int i = 0; i < outputBuffer.rows(); i++) {
		Eigen::RowVector3d point = Eigen::RowVector3d(inputCloud->points[i].x, inputCloud->points[i].y, inputCloud->points[i].z);
		outputBuffer.row(i) = point;
	}
	return outputBuffer;
}

Eigen::MatrixXd transformPointsWithTransformMatrix(const Eigen::MatrixXd& inputVertices, const Eigen::Matrix4f& transformMatrix)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>());
	Eigen::MatrixXd outputMatrix;
	*sourceCloud = convertEigenMatrixXdToPCLCloud(inputVertices);
	pcl::transformPointCloud(*sourceCloud, *transformedCloud, transformMatrix);
	outputMatrix = convertPCLCloudToEigenMatrixXd(transformedCloud);
	return outputMatrix;
}

Eigen::Matrix4f transformVerticesFromPointToPoint(const Eigen::MatrixXd& targetVertices, const Eigen::Vector3d fromPoint, const Eigen::Vector3d toPoint, Eigen::MatrixXd& outPoints)
{
	Eigen::MatrixXd outputBuffer;
	Eigen::Vector3d diffVector = (toPoint - fromPoint);
	Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	transform(0, 3) = diffVector.x(); transform(1, 3) = diffVector.y(); transform(2, 3) = diffVector.z();
	outputBuffer = transformPointsWithTransformMatrix(targetVertices, transform);
	outPoints = outputBuffer;
	return transform;
}

void transformPointCloudToOriginal(pcl::PointCloud<pcl::PointXYZ>::Ptr& inCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& outCloud)
{
	const Eigen::MatrixXd cloudMatrix = convertPCLCloudToEigenMatrixXd(inCloud);
	std::cout << "Initial targetVertical --------------------" << std::endl;
	Eigen::MatrixXd targetVertical = cloudMatrix;
	std::cout << "Initial fromPoints --------------------" << std::endl;
	Eigen::Vector3d fromPoints = cloudMatrix.colwise().mean();
	std::cout << "Initial toPoints --------------------" << std::endl;
	Eigen::Vector3d toPoints(0, 0, 0);
	std::cout << "Initial outPoints --------------------" << std::endl;
	Eigen::MatrixXd outPoints;
	std::cout << "Calculate transformation matrix ---------------" << std::endl;
	Eigen::Matrix4f transformMat = transformVerticesFromPointToPoint(targetVertical, fromPoints, toPoints, outPoints);

	*outCloud = convertEigenMatrixXdToPCLCloud(outPoints);
}

void readCameraMatrix(std::string infoYMLFilePath, std::vector<double>& infoVec, std::string obj_name)
{
	YAML::Node config_obj_29 = YAML::LoadFile(infoYMLFilePath);
	for (std::size_t i = 0; i < 9; i++)
	{
		infoVec.push_back(config_obj_29[obj_name]["cam_K"][i].as<double>());
	}
}

void configBoundingBox(std::string gtFilePath, std::vector<double>& boundingBoxVec, std::string obj_name, int className)
{
	YAML::Node config_obj_29 = YAML::LoadFile(gtFilePath);

	for (auto f : config_obj_29[obj_name])
	{
		boundingBoxVec.push_back(className);
		boundingBoxVec.push_back((f["obj_bb"][0].as<double>() + f["obj_bb"][2].as<double>() / 2) / 400);
		boundingBoxVec.push_back((f["obj_bb"][1].as<double>() + f["obj_bb"][3].as<double>() / 2) / 400);
		boundingBoxVec.push_back(f["obj_bb"][2].as<double>() / 400);
		boundingBoxVec.push_back(f["obj_bb"][3].as<double>() / 400);
	}
}

void removeLeadingZeros(std::string& str)
{
	const std::regex pattern("^0+(?!$)");
	str = regex_replace(str, pattern, "");
}
/*--------------MAIN PROCESSING FUNCTION----------------*/
void datasetGeneration()
{
	std::cout << "datasetGeneration:: Initialization" << std::endl;

	std::string inputFolder = "D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/datasetGeneration/Inputs";
	std::string outputFolder = "D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/datasetGeneration/Outputs";
	std::string debugFolder = "D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/datasetGeneration/Debugs";

	std::string obj_29_inputFolder = inputFolder + "/29";
	std::string obj_30_inputFolder = inputFolder + "/30";

	std::string obj_29_outputFolder = outputFolder + "/29";
	std::string obj_30_outputFolder = outputFolder + "/30";

	std::cout << "datasetGeneration:: Execution" << std::endl;
	std::cout << "datasetGeneration:: Execution: Extract bounding box" << std::endl;

	std::cout << "datasetGeneration:: Execution: Extract bounding box: Class 0" << std::endl;

	int className = 0;
	std::size_t n = 4;
	for (int obj_name = 0; obj_name < 1296; obj_name++)
	{
		std::vector<double>bbMatrix;
		std::string obj_name_str = std::to_string(obj_name);
		configBoundingBox(obj_29_inputFolder + "/gt.yml", bbMatrix, obj_name_str, className);
		std::fstream saveFile;
		int precision = n - std::min(n, obj_name_str.size());
		obj_name_str.insert(0, precision, '0');
		saveFile.open(obj_29_outputFolder + "/" + obj_name_str + ".txt", std::ios_base::out);
		for (int i = 0; i < bbMatrix.size(); i++)
		{
			saveFile << bbMatrix[i] << " ";
		}
		std::cout << "datasetGeneration:: Execution: Extract bounding box: Class 0: " + obj_name_str << std::endl;
	}

	std::cout << "datasetGeneration:: Execution: Extract bounding box: Class 1" << std::endl;

	className = 1;
	for (int obj_name = 0; obj_name < 1296; obj_name++)
	{
		std::vector<double>bbMatrix;
		std::string obj_name_str = std::to_string(obj_name);
		configBoundingBox(obj_30_inputFolder + "/gt.yml", bbMatrix, obj_name_str, className);
		std::fstream saveFile;
		int precision = n - std::min(n, obj_name_str.size());
		obj_name_str.insert(0, precision, '0');
		saveFile.open(obj_30_outputFolder + "/" + obj_name_str + ".txt", std::ios_base::out);
		for (int i = 0; i < bbMatrix.size(); i++)
		{
			saveFile << bbMatrix[i] << " ";
		}
		std::cout << "datasetGeneration:: Execution: Extract bounding box: Class 1: " + obj_name_str << std::endl;
	}
	std::cout << "datasetGeneration:: Finalization" << std::endl;
}

void detectMultipleObjects()
{
	std::cout << "detectMultipleObjects:: Initialization" << std::endl;

	std::string inputFolder = "D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/detectMultipleObjects/Inputs";
	std::string outputFolder = "D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/detectMultipleObjects/Outputs";
	std::string debugFolder = "D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/detectMultipleObjects/Debugs";

	std::string obj_0_inputFolder = inputFolder + "/0/*.jpg";
	std::string obj_1_inputFolder = inputFolder + "/1/*.jpg";

	std::vector<std::string> obj_0_inputFileNames;
	std::vector<std::string> obj_1_inputFileNames;

	std::size_t index = 0;

	std::cout << "detectMultipleObjects:: Execution" << std::endl;
	std::cout << "detectMultipleObjects:: Execution: Merge two object into one images" << std::endl;

	cv::glob(obj_0_inputFolder, obj_0_inputFileNames);
	cv::glob(obj_1_inputFolder, obj_1_inputFileNames);

	std::sort(obj_0_inputFileNames.begin(), obj_0_inputFileNames.end(), naturalSorting);
	std::sort(obj_1_inputFileNames.begin(), obj_1_inputFileNames.end(), naturalSorting);

	for (auto const& f : obj_0_inputFileNames)
	{
		cv::Mat obj_0_rgb = cv::imread(obj_0_inputFileNames[index]);
		cv::Mat obj_1_rgb = cv::imread(obj_1_inputFileNames[index]);

		std::string obj_0_nameFile = obj_0_inputFileNames[index].substr(obj_0_inputFileNames[index].find("\\") + 1);
		std::string toErase = ".jpg";
		eraseSubStrings(obj_0_nameFile, toErase);

		std::string obj_1_nameFile = obj_1_inputFileNames[index].substr(obj_1_inputFileNames[index].find("\\") + 1);
		eraseSubStrings(obj_1_nameFile, toErase);

		cv::Mat obj_1_merge_obj_2_h, obj_1_merge_obj_2_v;
		cv::hconcat(obj_0_rgb, obj_1_rgb, obj_1_merge_obj_2_h);
		cv::vconcat(obj_0_rgb, obj_1_rgb, obj_1_merge_obj_2_v);

		std::string savePathMergeH = outputFolder + "/" + obj_0_nameFile + "_" + obj_1_nameFile + "_h.jpg";
		std::string savePathMergev = outputFolder + "/" + obj_0_nameFile + "_" + obj_1_nameFile + "_v.jpg";

		cv::imwrite(savePathMergeH, obj_1_merge_obj_2_h);
		cv::imwrite(savePathMergev, obj_1_merge_obj_2_v);

		index++;
	}

	std::cout << "detectMultipleObjects:: Execution: Merge two object into one images: Success" << std::endl;
	std::cout << "detectMultipleObjects:: Finalization" << std::endl;
}

void cutPointCloudOfDetectObjects()
{
	std::cout << "cutPointCloudOfDetectObjects:: Initialization" << std::endl;

	std::string inputFolder = "D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/cutPointCloudOfDetectObjects/Inputs";
	std::string outputFolder = "D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/cutPointCloudOfDetectObjects/Outputs";
	std::string debugFolder = "D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/cutPointCloudOfDetectObjects/Debugs";

	std::string obj_0_depth = inputFolder + "/obj_0_depth/";
	std::string obj_1_depth = inputFolder + "/obj_1_depth/";

	std::string obj_0_rgb = inputFolder + "/obj_0_rgb/";
	std::string obj_1_rgb = inputFolder + "/obj_1_rgb/";

	std::string info_obj_0 = inputFolder + "/obj_0_info/info.yml";
	std::string gt_obj_0 = inputFolder + "/obj_0_info/gt.yml";

	std::string info_obj_1 = inputFolder + "/obj_1_info/info.yml";
	std::string gt_obj_1 = inputFolder + "/obj_1_info/gt.yml";

	std::string boundingBox = inputFolder + "/boundingBox/";
	std::string mergeDepthImage = inputFolder + "/mergeDepthImage/";

	std::string obj_0_depth_output = outputFolder + "/cropDepthImage/0";
	std::string obj_1_depth_output = outputFolder + "/cropDepthImage/1";

	std::vector<std::string> obj_0_depthFileNames;
	std::vector<std::string> obj_1_depthFileNames;

	std::vector<std::string> obj_0_rgbFileNames;
	std::vector<std::string> obj_1_rgbFileNames;

	std::vector<std::string> boundingBoxFileNames;
	std::vector<std::string> mergeDepthImageFileNames;

	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_0_merge_h(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_0_merge_v(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_1_merge_h(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr obj_1_merge_v(new pcl::PointCloud<pcl::PointXYZ>);

	std::string toEraseTXT = ".txt";
	std::string toErasePNG = ".png";

	bool CONCATE_DEPTH_IMAGE_FLAG = false;
	if (CONCATE_DEPTH_IMAGE_FLAG == true)
	{
		std::cout << "cutPointCloudOfDetectObjects:: Execution" << std::endl;
		std::cout << "cutPointCloudOfDetectObjects:: Execution: Concatenate depth images" << std::endl;
		std::size_t index = 0;

		cv::glob(obj_0_depth, obj_0_depthFileNames);
		cv::glob(obj_1_depth, obj_1_depthFileNames);

		cv::glob(obj_0_rgb, obj_0_rgbFileNames);
		cv::glob(obj_1_rgb, obj_1_rgbFileNames);

		std::sort(obj_0_depthFileNames.begin(), obj_0_depthFileNames.end(), naturalSorting);
		std::sort(obj_1_depthFileNames.begin(), obj_1_depthFileNames.end(), naturalSorting);

		std::sort(obj_0_rgbFileNames.begin(), obj_0_rgbFileNames.end(), naturalSorting);
		std::sort(obj_1_rgbFileNames.begin(), obj_1_rgbFileNames.end(), naturalSorting);

		for (auto const& f : obj_0_depthFileNames)
		{
			std::size_t obj_0_depth_index = obj_0_depthFileNames[index].find("\\") + 1;
			std::size_t obj_1_depth_index = obj_1_depthFileNames[index].find("\\") + 1;

			std::size_t obj_0_rgb_index = obj_0_rgbFileNames[index].find("\\") + 1;
			std::size_t obj_1_rgb_index = obj_1_rgbFileNames[index].find("\\") + 1;

			cv::Mat obj_0_depth_img = cv::imread(obj_0_depth + "/" + obj_0_depthFileNames[index].substr(obj_0_depth_index, 4) + ".png", cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
			cv::Mat obj_1_depth_img = cv::imread(obj_1_depth + "/" + obj_1_depthFileNames[index].substr(obj_1_depth_index, 4) + ".png", cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

			cv::Mat obj_0_rgb_img = cv::imread(obj_0_rgb + "/" + obj_0_rgbFileNames[index].substr(obj_0_rgb_index, 4) + ".png");
			cv::Mat obj_1_rgb_img = cv::imread(obj_1_rgb + "/" + obj_1_rgbFileNames[index].substr(obj_1_rgb_index, 4) + ".png");

			cv::Mat obj_0_merge_obj_1_depth_h, obj_0_merge_obj_1_depth_v;
			cv::Mat obj_0_merge_obj_1_rgb_h, obj_0_merge_obj_1_rgb_v;

			cv::hconcat(obj_0_depth_img, obj_1_depth_img, obj_0_merge_obj_1_depth_h);
			cv::hconcat(obj_0_rgb_img, obj_1_rgb_img, obj_0_merge_obj_1_rgb_h);

			cv::vconcat(obj_0_depth_img, obj_1_depth_img, obj_0_merge_obj_1_depth_v);
			cv::vconcat(obj_0_rgb_img, obj_1_rgb_img, obj_0_merge_obj_1_rgb_v);

			std::string savePathMergeDepthH = outputFolder + "/depth/" + obj_0_depthFileNames[index].substr(obj_0_depth_index, 4) + "_" + obj_1_depthFileNames[index].substr(obj_1_depth_index, 4) + "_h.png";
			std::string savePathMergeDepthV = outputFolder + "/depth/" + obj_0_depthFileNames[index].substr(obj_0_depth_index, 4) + "_" + obj_1_depthFileNames[index].substr(obj_1_depth_index, 4) + "_v.png";

			std::string savePathMergeRGBH = outputFolder + "/rgb/" + obj_0_rgbFileNames[index].substr(obj_0_rgb_index, 4) + "_" + obj_1_rgbFileNames[index].substr(obj_1_rgb_index, 4) + "_h.png";
			std::string savePathMergeRGBV = outputFolder + "/rgb/" + obj_0_rgbFileNames[index].substr(obj_0_rgb_index, 4) + "_" + obj_1_rgbFileNames[index].substr(obj_1_rgb_index, 4) + "_v.png";

			cv::imwrite(savePathMergeDepthH, obj_0_merge_obj_1_depth_h);
			cv::imwrite(savePathMergeDepthV, obj_0_merge_obj_1_depth_v);

			cv::imwrite(savePathMergeRGBH, obj_0_merge_obj_1_rgb_h);
			cv::imwrite(savePathMergeRGBV, obj_0_merge_obj_1_rgb_v);

			index++;
		}
		std::cout << "cutPointCloudOfDetectObjects:: Execution: Concatenate depth images: Success" << std::endl;
	}

	bool CUT_POINT_CLOUD_FLAG = true;


	if (CUT_POINT_CLOUD_FLAG == true)
	{
		std::cout << "cutPointCloudOfDetectObjects:: Execution" << std::endl;
		std::cout << "cutPointCloudOfDetectObjects:: Execution: Cut point cloud from bounding box" << std::endl;

		std::size_t index = 0;
		std::size_t count = 0;

		cv::glob(obj_0_depth, obj_0_depthFileNames);
		cv::glob(obj_1_depth, obj_1_depthFileNames);
		cv::glob(boundingBox, boundingBoxFileNames);
		cv::glob(mergeDepthImage, mergeDepthImageFileNames);

		std::sort(obj_0_depthFileNames.begin(), obj_0_depthFileNames.end(), naturalSorting);
		std::sort(obj_1_depthFileNames.begin(), obj_1_depthFileNames.end(), naturalSorting);
		std::sort(boundingBoxFileNames.begin(), boundingBoxFileNames.end(), naturalSorting);
		std::sort(mergeDepthImageFileNames.begin(), mergeDepthImageFileNames.end(), naturalSorting);

		for (auto const& f : boundingBoxFileNames)
		{
			std::ifstream inputBoundingBox(boundingBoxFileNames[index]);
			double boundingBoxArr[2][5];
			if (!inputBoundingBox.is_open())
			{
				std::cout << "Error opening file" << std::endl;
			}
			for (std::size_t r = 0; r < 2; r++)
			{
				for (std::size_t c = 0; c < 5; c++)
				{
					inputBoundingBox >> boundingBoxArr[r][c];
				}
			}

			std::size_t indexBB = boundingBoxFileNames[index].find("\\") + 1;
			std::string obj_0_num = boundingBoxFileNames[index].substr(indexBB, 4);
			std::string obj_1_num = boundingBoxFileNames[index].substr(indexBB + 5, 4);
			std::string checkHorizontalOrVertical = boundingBoxFileNames[index].substr(indexBB + 10, 1);

			if (checkHorizontalOrVertical == "h")
			{
				for (std::size_t r = 0; r < 2; r++)
				{
					boundingBoxArr[r][1] *= 800;
					boundingBoxArr[r][2] *= 400;
					boundingBoxArr[r][3] *= 800;
					boundingBoxArr[r][4] *= 400;

					boundingBoxArr[r][1] = boundingBoxArr[r][1] - boundingBoxArr[r][3] / 2;
					boundingBoxArr[r][2] = boundingBoxArr[r][2] - boundingBoxArr[r][4] / 2;
					boundingBoxArr[r][3] = boundingBoxArr[r][1] + boundingBoxArr[r][3];
					boundingBoxArr[r][4] = boundingBoxArr[r][2] + boundingBoxArr[r][4];

					if (boundingBoxArr[r][0] == 0)
					{
						cv::Mat obj_0_depth_img = cv::imread(obj_0_depth + obj_0_num + ".png", cv::IMREAD_ANYDEPTH);
						double boundingBox[4] = { boundingBoxArr[r][1], boundingBoxArr[r][2], boundingBoxArr[r][3], boundingBoxArr[r][4] };
						cropTheObjectFromDepthImage(obj_0_depth_img, boundingBox);
						
						std::string pathNum = obj_0_num;
						std::vector<double>infoMatrix;
						removeLeadingZeros(obj_0_num);
						readCameraMatrix(info_obj_0, infoMatrix, obj_0_num);
						double intrinsic[4] = { infoMatrix.at(0), infoMatrix.at(4), infoMatrix.at(2), infoMatrix.at(5) };

						pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = generatePointCloudFromDepthImage(obj_0_depth_img, intrinsic);
						pcl::io::savePLYFile(obj_0_depth_output + "/" + pathNum + "_h.ply", *cloud);
						std::cout << obj_0_depth_output + "/" + pathNum + "_h.ply" << std::endl;
					}
					else if (boundingBoxArr[r][0] == 1)
					{
						cv::Mat obj_1_depth_img = cv::imread(obj_1_depth + obj_1_num + ".png", cv::IMREAD_ANYDEPTH);
						double boundingBox[4] = { boundingBoxArr[r][1] - 400, boundingBoxArr[r][2], boundingBoxArr[r][3] - 400, boundingBoxArr[r][4] };
						cropTheObjectFromDepthImage(obj_1_depth_img, boundingBox);

						std::string pathNum = obj_1_num;
						std::vector<double>infoMatrix;
						removeLeadingZeros(obj_1_num);
						readCameraMatrix(info_obj_1, infoMatrix, obj_1_num);
						double intrinsic[4] = { infoMatrix.at(0), infoMatrix.at(4), infoMatrix.at(2), infoMatrix.at(5) };

						pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = generatePointCloudFromDepthImage(obj_1_depth_img, intrinsic);
						pcl::io::savePLYFile(obj_1_depth_output + "/" + pathNum + "_h.ply", *cloud);
						std::cout << obj_1_depth_output + "/" + pathNum + "_h.ply" << std::endl;


					}
				}
			}
			else if (checkHorizontalOrVertical == "v")
			{
				for (std::size_t r = 0; r < 2; r++)
				{
					boundingBoxArr[r][1] *= 400;
					boundingBoxArr[r][2] *= 800;
					boundingBoxArr[r][3] *= 400;
					boundingBoxArr[r][4] *= 800;

					boundingBoxArr[r][1] = boundingBoxArr[r][1] - boundingBoxArr[r][3] / 2;
					boundingBoxArr[r][2] = boundingBoxArr[r][2] - boundingBoxArr[r][4] / 2;
					boundingBoxArr[r][3] = boundingBoxArr[r][1] + boundingBoxArr[r][3];
					boundingBoxArr[r][4] = boundingBoxArr[r][2] + boundingBoxArr[r][4];

					if (boundingBoxArr[r][0] == 0)
					{
						cv::Mat obj_0_depth_img = cv::imread(obj_0_depth + obj_0_num + ".png", cv::IMREAD_ANYDEPTH);
						double boundingBox[4] = { boundingBoxArr[r][1], boundingBoxArr[r][2], boundingBoxArr[r][3], boundingBoxArr[r][4] };
						cropTheObjectFromDepthImage(obj_0_depth_img, boundingBox);

						std::string pathNum = obj_0_num;
						std::vector<double>infoMatrix;
						removeLeadingZeros(obj_0_num);
						readCameraMatrix(info_obj_0, infoMatrix, obj_0_num);
						double intrinsic[4] = { infoMatrix.at(0), infoMatrix.at(4), infoMatrix.at(2), infoMatrix.at(5) };

						pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = generatePointCloudFromDepthImage(obj_0_depth_img, intrinsic);
						pcl::io::savePLYFile(obj_0_depth_output + "/" + pathNum + "_v.ply", *cloud);
						std::cout << obj_0_depth_output + "/" + pathNum + "_v.ply" << std::endl;
					}
					else if (boundingBoxArr[r][0] == 1)
					{
						cv::Mat obj_1_depth_img = cv::imread(obj_1_depth + obj_1_num + ".png", cv::IMREAD_ANYDEPTH);
						double boundingBox[4] = { boundingBoxArr[r][1], boundingBoxArr[r][2] - 400, boundingBoxArr[r][3], boundingBoxArr[r][4] - 400 };
						cropTheObjectFromDepthImage(obj_1_depth_img, boundingBox);

						std::string pathNum = obj_1_num;
						std::vector<double>infoMatrix;
						removeLeadingZeros(obj_1_num);
						readCameraMatrix(info_obj_1, infoMatrix, obj_1_num);
						double intrinsic[4] = { infoMatrix.at(0), infoMatrix.at(4), infoMatrix.at(2), infoMatrix.at(5) };

						pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = generatePointCloudFromDepthImage(obj_1_depth_img, intrinsic);
						pcl::io::savePLYFile(obj_1_depth_output + "/" + pathNum + "_v.ply", *cloud);
						std::cout << obj_1_depth_output + "/" + pathNum + "_v.ply" << std::endl;
					}
				}
			}
			index++;
		}
		std::cout << "cutPointCloudOfDetectObjects:: Execution: Cut point cloud from bounding box: Success" << std::endl;
	}
}

void mergePointClouds()
{
	std::cout << "mergePointClouds:: Initialization" << std::endl;

	std::string inputFolder = "D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/mergePointClouds/Inputs";
	std::string outputFolder = "D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/mergePointClouds/Outputs";
	std::string debugFolder = "D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/mergePointClouds/Debugs";

	std::string obj_0_folder = inputFolder + "/0/";
	std::string obj_1_folder = inputFolder + "/1/";

	std::string obj_0_info_file = inputFolder + "/obj_0_info/info.yml";
	std::string obj_1_info_file = inputFolder + "/obj_1_info/info.yml";

	std::string obj_0_gt_file = inputFolder + "/obj_0_info/gt.yml";
	std::string obj_1_gt_file = inputFolder + "/obj_1_info/gt.yml";

	std::vector<std::string> obj_0_fileNames;
	std::vector<std::string> obj_1_fileNames;

	std::size_t index = 0;

	std::cout << "mergePointClouds:: Execution" << std::endl;
	cv::glob(obj_0_folder, obj_0_fileNames);
	cv::glob(obj_1_folder, obj_1_fileNames);

	std::sort(obj_0_fileNames.begin(), obj_0_fileNames.end(), naturalSorting);
	std::sort(obj_1_fileNames.begin(), obj_1_fileNames.end(), naturalSorting);

	pcl::PointCloud<pcl::PointXYZ>::Ptr mergePointCloud_obj_0_h(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr mergePointCloud_obj_0_v(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::PointCloud<pcl::PointXYZ>::Ptr mergePointCloud_obj_1_h(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr mergePointCloud_obj_1_v(new pcl::PointCloud<pcl::PointXYZ>);


	for (auto const& f : obj_0_fileNames)
	{
		std::size_t indexBB = obj_0_fileNames[index].find("\\") + 1;
		std::string obj_0_num = obj_0_fileNames[index].substr(indexBB, 4);

		std::string checkHorizontalOrVertical = obj_0_fileNames[index].substr(indexBB + 5, 1);

		if (checkHorizontalOrVertical == "h")
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::io::loadPLYFile(obj_0_folder + obj_0_num + "_h.ply", *cloud);
			Eigen::Matrix4f transformMat;
			removeLeadingZeros(obj_0_num);
			YAML::Node obj_0_gt = YAML::LoadFile(obj_0_gt_file);
			for (auto f : obj_0_gt[obj_0_num])
			{
				transformMat << f["cam_R_m2c"][0].as<double>(), f["cam_R_m2c"][1].as<double>(), f["cam_R_m2c"][2].as<double>(), f["cam_t_m2c"][0].as<double>(),
					f["cam_R_m2c"][3].as<double>(), f["cam_R_m2c"][4].as<double>(), f["cam_R_m2c"][5].as<double>(), f["cam_t_m2c"][1].as<double>(),
					f["cam_R_m2c"][6].as<double>(), f["cam_R_m2c"][7].as<double>(), f["cam_R_m2c"][8].as<double>(), f["cam_t_m2c"][2].as<double>(),
					0, 0, 0, 1;
			}

			pcl::PointCloud<pcl::PointXYZ>::Ptr transformedPointCloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::transformPointCloud(*cloud, *transformedPointCloud, transformMat.inverse());
			*mergePointCloud_obj_0_h += *transformedPointCloud;
			pcl::io::savePLYFile(outputFolder + "/0/h/" + obj_0_num + ".ply", *transformedPointCloud);
			std::cout << outputFolder + "/0/h/" + obj_0_num + ".ply" << std::endl;
		}
		else if (checkHorizontalOrVertical == "v")
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::io::loadPLYFile(obj_0_folder + obj_0_num + "_h.ply", *cloud);
			Eigen::Matrix4f transformMat;
			removeLeadingZeros(obj_0_num);
			YAML::Node obj_0_gt = YAML::LoadFile(obj_0_gt_file);
			for (auto f : obj_0_gt[obj_0_num])
			{
				transformMat << f["cam_R_m2c"][0].as<double>(), f["cam_R_m2c"][1].as<double>(), f["cam_R_m2c"][2].as<double>(), f["cam_t_m2c"][0].as<double>(),
					f["cam_R_m2c"][3].as<double>(), f["cam_R_m2c"][4].as<double>(), f["cam_R_m2c"][5].as<double>(), f["cam_t_m2c"][1].as<double>(),
					f["cam_R_m2c"][6].as<double>(), f["cam_R_m2c"][7].as<double>(), f["cam_R_m2c"][8].as<double>(), f["cam_t_m2c"][2].as<double>(),
					0, 0, 0, 1;
			}
			pcl::PointCloud<pcl::PointXYZ>::Ptr transformedPointCloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::transformPointCloud(*cloud, *transformedPointCloud, transformMat.inverse());
			*mergePointCloud_obj_0_v += *transformedPointCloud;
			pcl::io::savePLYFile(outputFolder + "/0/v/" + obj_0_num + ".ply", *transformedPointCloud);
			std::cout << outputFolder + "/0/v/" + obj_0_num + ".ply" << std::endl;
		}
		index++;
	}

	index = 0;

	for (auto const& f : obj_1_fileNames)
	{
		std::size_t indexBB = obj_1_fileNames[index].find("\\") + 1;
		std::string obj_1_num = obj_1_fileNames[index].substr(indexBB, 4);

		std::string checkHorizontalOrVertical = obj_1_fileNames[index].substr(indexBB + 5, 1);

		if (checkHorizontalOrVertical == "h")
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::io::loadPLYFile(obj_1_folder + obj_1_num + "_h.ply", *cloud);
			Eigen::Matrix4f transformMat;
			removeLeadingZeros(obj_1_num);
			YAML::Node obj_1_gt = YAML::LoadFile(obj_1_gt_file);
			for (auto f : obj_1_gt[obj_1_num])
			{
				transformMat << f["cam_R_m2c"][0].as<double>(), f["cam_R_m2c"][1].as<double>(), f["cam_R_m2c"][2].as<double>(), f["cam_t_m2c"][0].as<double>(),
					f["cam_R_m2c"][3].as<double>(), f["cam_R_m2c"][4].as<double>(), f["cam_R_m2c"][5].as<double>(), f["cam_t_m2c"][1].as<double>(),
					f["cam_R_m2c"][6].as<double>(), f["cam_R_m2c"][7].as<double>(), f["cam_R_m2c"][8].as<double>(), f["cam_t_m2c"][2].as<double>(),
					0, 0, 0, 1;
			}

			pcl::PointCloud<pcl::PointXYZ>::Ptr transformedPointCloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::transformPointCloud(*cloud, *transformedPointCloud, transformMat.inverse());
			*mergePointCloud_obj_1_h += *transformedPointCloud;
			pcl::io::savePLYFile(outputFolder + "/1/h/" + obj_1_num + ".ply", *transformedPointCloud);
			std::cout << outputFolder + "/1/h/" + obj_1_num + ".ply" << std::endl;
		}
		else if (checkHorizontalOrVertical == "v")
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::io::loadPLYFile(obj_1_folder + obj_1_num + "_h.ply", *cloud);
			Eigen::Matrix4f transformMat;
			removeLeadingZeros(obj_1_num);
			YAML::Node obj_1_gt = YAML::LoadFile(obj_1_gt_file);
			for (auto f : obj_1_gt[obj_1_num])
			{
				transformMat << f["cam_R_m2c"][0].as<double>(), f["cam_R_m2c"][1].as<double>(), f["cam_R_m2c"][2].as<double>(), f["cam_t_m2c"][0].as<double>(),
					f["cam_R_m2c"][3].as<double>(), f["cam_R_m2c"][4].as<double>(), f["cam_R_m2c"][5].as<double>(), f["cam_t_m2c"][1].as<double>(),
					f["cam_R_m2c"][6].as<double>(), f["cam_R_m2c"][7].as<double>(), f["cam_R_m2c"][8].as<double>(), f["cam_t_m2c"][2].as<double>(),
					0, 0, 0, 1;
			}
			pcl::PointCloud<pcl::PointXYZ>::Ptr transformedPointCloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::transformPointCloud(*cloud, *transformedPointCloud, transformMat.inverse());
			*mergePointCloud_obj_1_v += *transformedPointCloud;
			pcl::io::savePLYFile(outputFolder + "/1/v/" + obj_1_num + ".ply", *transformedPointCloud);
			std::cout << outputFolder + "/1/v/" + obj_1_num + ".ply" << std::endl;
		}
		index++;
	}
	pcl::io::savePLYFile(outputFolder + "/obj_0_h.ply", *mergePointCloud_obj_0_h);
	pcl::io::savePLYFile(outputFolder + "/obj_0_v.ply", *mergePointCloud_obj_0_v);
	pcl::io::savePLYFile(outputFolder + "/obj_1_h.ply", *mergePointCloud_obj_1_h);
	pcl::io::savePLYFile(outputFolder + "/obj_1_v.ply", *mergePointCloud_obj_1_v);
	std::cout << "mergePointClouds:: Execution: Success" << std::endl;
	std::cout << "mergePointClouds:: Finalization" << std::endl;

}

void test_func()
{
	/*double intrinsic_0[4] = { 1075.65091572 , 1073.90347929 ,213.06888344, 175.72159802 };
	double intrinsic_1[4] = { 1075.65091572 , 1073.90347929 ,171.72159802, 175.72159802 };
	double boundingBox_0[4] = { 100, 140, 300, 270 };
	double boundingBox_1[4] = { 95, 130, 300, 274 };

	Eigen::Matrix4f transform_0;
	Eigen::Matrix4f transform_1;

	transform_0 << 0.99978957, 0.01971206, -0.00566635, -7.34830894, 0.01885930, -0.99212728, -0.12380646, 13.93064806, -0.00806225, 0.12367446, -0.99228976, 628.08193448, 0, 0, 0, 1;
	transform_1 << 0.99689798, 0.07814465, 0.00934223, -7.45212185, -0.07870095, 0.99003960, 0.11673884, 16.40548422, -0.00012667, -0.11711129, 0.99311935, 628.60539089, 0, 0, 0, 1;

	cv::Mat depth_0 = cv::imread("D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/test_func/Inputs/0000.png", cv::IMREAD_ANYDEPTH);
	cv::Mat depth_1 = cv::imread("D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/test_func/Inputs/1295.png", cv::IMREAD_ANYDEPTH);

	for (int x = 0; x < depth_0.rows; x++) {
		for (int y = 0; y < depth_0.cols; y++) {

			if (!(x >= 140 && x <= 270 && y >= 100 && y <= 300))
			{
				depth_0.ptr<ushort>(x)[y] = 0;

			}
		}
	}*/



	pcl::PLYReader reader;
	pcl::PassThrough<pcl::PointXYZ> pass;
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
	pcl::PCDWriter writer;
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	pcl::ExtractIndices<pcl::Normal> extract_normals;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());


	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered2(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
	pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients), coefficients_cylinder(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices), inliers_cylinder(new pcl::PointIndices);

	reader.read("D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/test_func/Inputs/0000_h.ply", *cloud);
	std::cerr << "PointCloud has: " << cloud->size() << " data points." << std::endl;

	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0, 1.5);
	pass.filter(*cloud_filtered);
	std::cerr << "PointCloud after filtering has: " << cloud_filtered->size() << " data points." << std::endl;

	ne.setSearchMethod(tree);
	ne.setInputCloud(cloud_filtered);
	ne.setKSearch(50);
	ne.compute(*cloud_normals);


	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
	seg.setNormalDistanceWeight(0.1);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(100);
	seg.setDistanceThreshold(0.03);
	seg.setInputCloud(cloud_filtered);
	seg.setInputNormals(cloud_normals);
	// Obtain the plane inliers and coefficients
	seg.segment(*inliers_plane, *coefficients_plane);
	std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

	extract.setInputCloud(cloud_filtered);
	extract.setIndices(inliers_plane);
	extract.setNegative(false);

	// Write the planar inliers to disk
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
	extract.filter(*cloud_plane);
	std::cerr << "PointCloud representing the planar component: " << cloud_plane->size() << " data points." << std::endl;
	writer.write("D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/test_func/table_scene_mug_stereo_textured_plane.ply", *cloud_plane, false);

	// Remove the planar inliers, extract the rest
	extract.setNegative(true);
	extract.filter(*cloud_filtered2);
	extract_normals.setNegative(true);
	extract_normals.setInputCloud(cloud_normals);
	extract_normals.setIndices(inliers_plane);
	extract_normals.filter(*cloud_normals2);

	// Create the segmentation object for cylinder segmentation and set all the parameters
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_CYLINDER);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setNormalDistanceWeight(0.1);
	seg.setMaxIterations(10000);
	seg.setDistanceThreshold(0.05);
	seg.setRadiusLimits(0, 0.1);
	seg.setInputCloud(cloud_filtered2);
	seg.setInputNormals(cloud_normals2);

	// Obtain the cylinder inliers and coefficients
	seg.segment(*inliers_cylinder, *coefficients_cylinder);
	std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

	// Write the cylinder inliers to disk
	extract.setInputCloud(cloud_filtered2);
	extract.setIndices(inliers_cylinder);
	extract.setNegative(false);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cylinder(new pcl::PointCloud<pcl::PointXYZ>());
	extract.filter(*cloud_cylinder);
	if (cloud_cylinder->points.empty())
		std::cerr << "Can't find the cylindrical component." << std::endl;
	else
	{
		std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->size() << " data points." << std::endl;
		writer.write("D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/test_func/table_scene_mug_stereo_textured_cylinder.ply", *cloud_cylinder, false);
	}

}


/*--------------MAIN FUNCTIONS--------------------------*/
void mainFunction()
{
	std::cout << "mainFunction:: Initialization" << std::endl;
	std::cout << "mainFunction:: Execution" << std::endl;
	test_func();
	std::cout << "mainFunction:: Finalization" << std::endl;
}