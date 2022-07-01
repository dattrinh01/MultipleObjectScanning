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

pcl::PointCloud<pcl::PointXYZ>::Ptr createPointCloud(cv::Mat depth_img, const double depth_intrinsic[4])
{
	const double fx = depth_intrinsic[0];
	const double fy = depth_intrinsic[1];
	const double cx = depth_intrinsic[2];
	const double cy = depth_intrinsic[3];

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	for (int x = 0; x < depth_img.rows; x++) {
		for (int y = 0; y < depth_img.cols; y++) {
			pcl::PointXYZ p;

			double depth_val = depth_img.ptr<ushort>(x)[y];
			if (depth_val == 0) { continue; }
			p.z = depth_val;
			p.x = (y - cx) * p.z / fx;
			p.y = (x - cy) * p.z / fy;
			cloud->points.push_back(p);

		}
	}
	return cloud;
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

pcl::PointCloud<pcl::PointXYZ>::Ptr generatePointCloudFromDepthImage(cv::Mat depth_img, const double depth_intrinsic[4])
{
	const double fx = depth_intrinsic[0];
	const double fy = depth_intrinsic[1];
	const double cx = depth_intrinsic[2];
	const double cy = depth_intrinsic[3];

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	for (int x = 0; x < depth_img.rows; x++) {
		for (int y = 0; y < depth_img.cols; y++) {
			pcl::PointXYZ p;

			double depth_val = depth_img.ptr<ushort>(x)[y];
			if (depth_val == 0) { continue; }
			p.z = depth_val;
			p.x = (y - cx) * p.z / fx;
			p.y = (x - cy) * p.z / fy;
			cloud->points.push_back(p);
		}
	}
	return cloud;
}

void cropAndCreatePointCloud(std::string boundingBoxPath, std::string depthPath, std::string outputPath)
{
	std::ifstream inputfile(boundingBoxPath);
	double boundingBoxArr[2][5];
	if (!inputfile.is_open())
	{
		std::cout << "Error opening file";
	}
	for (int r = 0; r < 2; r++)
	{
		for (int c = 0; c < 5; c++)
		{
			inputfile >> boundingBoxArr[r][c];
		}
	}

	std::string checkVerticalOrHorizontal = boundingBoxPath.substr(boundingBoxPath.find(".") - 1);

	if (checkVerticalOrHorizontal == "v.txt")
	{
		boundingBoxArr[0][1] *= 1280;
		boundingBoxArr[0][2] *= 2048;
		boundingBoxArr[0][3] *= 1280;
		boundingBoxArr[0][4] *= 2048;

		boundingBoxArr[0][1] = boundingBoxArr[0][1] - boundingBoxArr[0][3] / 2;
		boundingBoxArr[0][2] = boundingBoxArr[0][2] - boundingBoxArr[0][4] / 2;
		boundingBoxArr[0][3] = boundingBoxArr[0][1] + boundingBoxArr[0][3];
		boundingBoxArr[0][4] = boundingBoxArr[0][2] + boundingBoxArr[0][4];

		boundingBoxArr[0][1] = (boundingBoxArr[0][1] / 1280 * 640) + 10;
		boundingBoxArr[0][2] = boundingBoxArr[0][2] / 2048 * 960;
		boundingBoxArr[0][3] = boundingBoxArr[0][3] / 1280 * 640;
		boundingBoxArr[0][4] = boundingBoxArr[0][4] / 2048 * 960;

		boundingBoxArr[1][1] *= 1280;
		boundingBoxArr[1][2] *= 2048;
		boundingBoxArr[1][3] *= 1280;
		boundingBoxArr[1][4] *= 2048;

		boundingBoxArr[1][1] = boundingBoxArr[1][1] - boundingBoxArr[1][3] / 2;
		boundingBoxArr[1][2] = boundingBoxArr[1][2] - boundingBoxArr[1][4] / 2;
		boundingBoxArr[1][3] = boundingBoxArr[1][1] + boundingBoxArr[1][3];
		boundingBoxArr[1][4] = boundingBoxArr[1][2] + boundingBoxArr[1][4];

		boundingBoxArr[1][1] = boundingBoxArr[1][1] / 1280 * 640;
		boundingBoxArr[1][2] = boundingBoxArr[1][2] / 2048 * 960;
		boundingBoxArr[1][3] = boundingBoxArr[1][3] / 1280 * 640;
		boundingBoxArr[1][4] = boundingBoxArr[1][4] / 2048 * 960;

		cv::Mat croppedImg1, croppedImg2;
		cv::Mat depth_frame = cv::imread(depthPath, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
		depth_frame(cv::Rect(boundingBoxArr[0][1] + 10, boundingBoxArr[0][2],
			(boundingBoxArr[0][3] - boundingBoxArr[0][1]) + 10, (boundingBoxArr[0][4] - boundingBoxArr[0][2]) + 10)).copyTo(croppedImg1);

		depth_frame(cv::Rect(boundingBoxArr[1][1] + 10, boundingBoxArr[1][2],
			(boundingBoxArr[1][3] - boundingBoxArr[1][1]) + 10, (boundingBoxArr[1][4] - boundingBoxArr[1][2]) + 10)).copyTo(croppedImg2);

		if (boundingBoxArr[0][0] == 0)
		{

		}
	}
	else
	{
		boundingBoxArr[0][1] *= 2560;
		boundingBoxArr[0][2] *= 1024;
		boundingBoxArr[0][3] *= 2560;
		boundingBoxArr[0][4] *= 1024;

		boundingBoxArr[0][1] = boundingBoxArr[0][1] - boundingBoxArr[0][3] / 2;
		boundingBoxArr[0][2] = boundingBoxArr[0][2] - boundingBoxArr[0][4] / 2;
		boundingBoxArr[0][3] = boundingBoxArr[0][1] + boundingBoxArr[0][3];
		boundingBoxArr[0][4] = boundingBoxArr[0][2] + boundingBoxArr[0][4];

		boundingBoxArr[0][1] = boundingBoxArr[0][1] / 2560 * 1280;
		boundingBoxArr[0][2] = boundingBoxArr[0][2] / 1024 * 480;
		boundingBoxArr[0][3] = boundingBoxArr[0][3] / 2560 * 1280;
		boundingBoxArr[0][4] = boundingBoxArr[0][4] / 1024 * 480;


		boundingBoxArr[1][1] *= 2560;
		boundingBoxArr[1][2] *= 1024;
		boundingBoxArr[1][3] *= 2560;
		boundingBoxArr[1][4] *= 1024;

		boundingBoxArr[1][1] = boundingBoxArr[1][1] - boundingBoxArr[1][3] / 2;
		boundingBoxArr[1][2] = boundingBoxArr[1][2] - boundingBoxArr[1][4] / 2;
		boundingBoxArr[1][3] = boundingBoxArr[1][1] + boundingBoxArr[1][3];
		boundingBoxArr[1][4] = boundingBoxArr[1][2] + boundingBoxArr[1][4];


		boundingBoxArr[1][1] = boundingBoxArr[1][1] / 2560 * 1280;
		boundingBoxArr[1][2] = boundingBoxArr[1][2] / 1024 * 480;
		boundingBoxArr[1][3] = boundingBoxArr[1][3] / 2560 * 1280;
		boundingBoxArr[1][4] = boundingBoxArr[1][4] / 1024 * 480;

		cv::Mat croppedImg;
		cv::Mat depth_frame = cv::imread(depthPath, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
		depth_frame(cv::Rect(boundingBoxArr[0][1], boundingBoxArr[0][2], boundingBoxArr[0][3], boundingBoxArr[0][4])).copyTo(croppedImg);
		cv::imwrite(boundingBoxPath + ".png", croppedImg);
		std::cout << "Save " + boundingBoxPath + ".png" << std::endl;
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

void cropImageHorizontalYOLOFormat(cv::Mat& originalDepthImage, std::string boundingBoxPath, int width, int height)
{
	cv::Mat cropDepthImage;
	std::ifstream inputBoundingBox(boundingBoxPath);
	double boundingBoxArr[2][5];
	if (!inputBoundingBox.is_open())
	{
		std::cout << "Error opening file" << std::endl;
	}
	else
	{
		for (std::size_t r = 0; r < 2; r++)
		{
			for (std::size_t c = 0; c < 5; c++)
			{
				inputBoundingBox >> boundingBoxArr[r][c];
			}
		}
	}

	for (std::size_t r = 0; r < 2; r++)
	{
		boundingBoxArr[r][1] *= width;
		boundingBoxArr[r][2] *= height;
		boundingBoxArr[r][3] *= width;
		boundingBoxArr[r][4] *= height;

		boundingBoxArr[r][1] = boundingBoxArr[r][1] - boundingBoxArr[r][3] / 2;
		boundingBoxArr[r][2] = boundingBoxArr[r][2] - boundingBoxArr[r][4] / 2;
		boundingBoxArr[r][3] = boundingBoxArr[r][1] + boundingBoxArr[r][3];
		boundingBoxArr[r][4] = boundingBoxArr[r][2] + boundingBoxArr[r][4];

		if (boundingBoxArr[r][0] == 0)
		{
			originalDepthImage(cv::Rect(boundingBoxArr[r][1], boundingBoxArr[r][2], boundingBoxArr[r][3] - boundingBoxArr[r][1], boundingBoxArr[r][4] - boundingBoxArr[r][2])).copyTo(cropDepthImage);
			cv::imwrite("D:/DATA/Research/DrNhu/MultipleObjectScanningDatasets/cutPointCloudOfDetectObjects/Outputs", cropDepthImage);
			std::cout << "Done" << std::endl;
		}
		else if (boundingBoxArr[r][0] == 1)
		{

		}
	}
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
	/*std::vector<double>infoMatrix;
	std::string obj_name = "0";
	readCameraMatrix(obj_29_inputFolder + "/info.yml", infoMatrix, obj_name);
	
	for (auto i : infoMatrix)
	{
		std::cout << i << " ";

	}*/

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

	std::string labelsFolder = inputFolder + "/labels/*.txt";
	std::string imagesFolder = inputFolder + "/rgb_images/*.jpg";
	std::string obj_0_depth = inputFolder + "/depth_images/0";
	std::string obj_1_depth = inputFolder + "/depth_images/1";
	std::string depthFolder = inputFolder + "/depth/*.png";

	std::string info_obj_0 = inputFolder + "/obj_0_info/info.yml";
	std::string gt_obj_0 = inputFolder + "/obj_0_info/gt.yml";
	
	std::string info_obj_1 = inputFolder + "/obj_1_info/info.yml";
	std::string gt_obj_1 = inputFolder + "/obj_1_info/gt.yml";

	std::vector<std::string> labelsFileNames;
	std::vector<std::string> imagesFileNames;
	std::vector<std::string> depthFileNames;

	std::size_t index = 0;
	std::string toEraseTXT = ".txt";
	std::string toErasePNG = ".png";

	int CONCATE_DEPTH_IMAGE_FLAG = false;

	if (CONCATE_DEPTH_IMAGE_FLAG == true)
	{
		std::cout << "cutPointCloudOfDetectObjects:: Execution" << std::endl;
		std::cout << "cutPointCloudOfDetectObjects:: Execution: Concatenate depth images" << std::endl;

		cv::glob(labelsFolder, labelsFileNames);

		std::sort(labelsFileNames.begin(), labelsFileNames.end(), naturalSorting);

		for (auto const& f : labelsFileNames)
		{
			std::string labelName = labelsFileNames[index];

			std::size_t index_ = labelName.find("_") + 1;

			cv::Mat obj_0_depth_img = cv::imread(obj_0_depth + "/" + labelName.substr(index_ - 5, 4) + ".png", cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
			cv::Mat obj_1_depth_img = cv::imread(obj_1_depth + "/" + labelName.substr(index_, 4) + ".png", cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

			cv::Mat obj_0_merge_obj_1_h, obj_0_merge_obj_1_v;
			cv::hconcat(obj_0_depth_img, obj_1_depth_img, obj_0_merge_obj_1_h);
			cv::vconcat(obj_0_depth_img, obj_1_depth_img, obj_0_merge_obj_1_v);

			std::string savePathMergeH = outputFolder + "/" + labelName.substr(index_ - 5, 4) + "_" + labelName.substr(index_, 4) + "_h.png";
			std::string savePathMergev = outputFolder + "/" + labelName.substr(index_ - 5, 4) + "_" + labelName.substr(index_, 4) + "_v.png";

			cv::imwrite(savePathMergeH, obj_0_merge_obj_1_h);
			cv::imwrite(savePathMergev, obj_0_merge_obj_1_v);

			index++;
		}
		std::cout << "cutPointCloudOfDetectObjects:: Execution: Concatenate depth images: Success" << std::endl;
	}

	index = 0;
	int CUT_POINT_CLOUD_FLAG = true;
	if (CUT_POINT_CLOUD_FLAG == true)
	{
		std::cout << "cutPointCloudOfDetectObjects:: Execution" << std::endl;
		std::cout << "cutPointCloudOfDetectObjects:: Execution: Cut Point Cloud Of Detect Objects" << std::endl;

		cv::glob(labelsFolder, labelsFileNames);
		cv::glob(depthFolder, depthFileNames);

		std::sort(labelsFileNames.begin(), labelsFileNames.end(), naturalSorting);
		std::sort(depthFileNames.begin(), depthFileNames.end(), naturalSorting);

		cv::Mat depth_img = cv::imread(debugFolder + "/0000_0001_h.png", cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);


		for (auto const& f : depthFileNames)
		{
			
			cv::Mat depth_img = cv::imread(depthFileNames[index], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

			std::cout << labelsFileNames[index] << std::endl;
			std::string labelPath;
			std::for_each(labelsFileNames.begin(), labelsFileNames.end(), [&](const std::string& piece) { labelPath += piece; });

			std::string checkHorizontalOrVertical = labelPath;
			eraseSubStrings(checkHorizontalOrVertical, toErasePNG);


			
			
			if (checkHorizontalOrVertical.back() == 'h')
			{
				cropImageHorizontalYOLOFormat(depth_img, labelPath, depth_img.size().width, depth_img.size().height);
			}
			index++;
		}

		std::cout << "cutPointCloudOfDetectObjects:: Execution: Cut Point Cloud Of Detect Objects: Success" << std::endl;
	}

	

	std::cout << "cutPointCloudOfDetectObjects:: Finalization" << std::endl;

}

/*--------------MAIN FUNCTIONS--------------------------*/
void mainFunction()
{
	std::cout << "mainFunction:: Initialization" << std::endl;
	std::cout << "mainFunction:: Execution" << std::endl;
	cutPointCloudOfDetectObjects();
	std::cout << "mainFunction:: Finalization" << std::endl;
}