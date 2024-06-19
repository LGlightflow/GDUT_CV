

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;
///////////////////////////////////////////
//exp1.1
//////////////////////////////////////////
cv::Mat sobel_x = (cv::Mat_<float>(3, 3) << -1, 0, 1,
	-2, 0, 2,
	-1, 0, 1);

// Sobel Y kernel (vertical edge detection)
cv::Mat sobel_y = (cv::Mat_<float>(3, 3) << -1, -2, -1,
	0, 0, 0,
	1, 2, 1);

// average smoothing kernel
cv::Mat averageKernel =(cv::Mat_<float>(3, 3) << 1.0 / 9, 1.0 / 9, 1.0 / 9,
							 1.0 / 9, 1.0 / 9, 1.0 / 9,
							 1.0 / 9, 1.0 / 9, 1.0 / 9);

// gaussian smoothing kernel
cv::Mat weightedAverageKernel = (cv::Mat_<float>(3, 3) << 1.0 / 16, 2.0 / 16, 1.0 / 16,
									 2.0 / 16, 4.0 / 16, 2.0 / 16,
									 1.0 / 16, 2.0 / 16, 1.0 / 16);

// sharppen kernel
cv::Mat lapalicanKernel = (cv::Mat_<float>(3, 3) << 0.0, -1.0, 0.0,
	-1.0, 5.0, -1.0,
	0.0, -1.0, 0.0);


// RGB to Y-component (YUV)
Mat getGrayImg(Mat img) {
	Mat gray(img.rows, img.cols, CV_8UC1, Scalar(0));
	Mat timg;
	img.convertTo(timg, CV_32F);

	for (int i = 0; i < timg.rows; i++) {
		for (int j = 0; j < timg.cols; j++) {
			// R*0.299 + G*0.587 + B*0.114
			float gray_intensity = timg.at<Vec3f>(i, j)[0] * 0.114 +timg.at<Vec3f>(i, j)[1] * 0.587 +timg.at<Vec3f>(i, j)[2] * 0.299;
			gray.at<uchar>(i, j) = round(gray_intensity);
		}
	}
	return gray;
}

Mat paddingWithZero(Mat img) {
	Mat padding_img(img.rows + 2, img.cols + 2, CV_8UC1, Scalar(0));
	for (int i = 1; i <= img.rows; i++) {
		for (int j = 1; j <= img.cols; j++) {
			padding_img.at<uchar>(i, j) = img.at<uchar>(i - 1, j - 1);
		}
	}
	return padding_img;
}

Mat paddingWithNeighbor(cv::Mat img) {
	cv::Mat padding_img(img.rows + 2, img.cols + 2, CV_8UC1, cv::Scalar(0));
	img.copyTo(padding_img(cv::Rect(1, 1, img.cols, img.rows)));
	for (int i = 1; i < img.rows + 1; ++i) {
		padding_img.at<uchar>(i, 0) = img.at<uchar>(i - 1, 0);
		padding_img.at<uchar>(i, img.cols + 1) = img.at<uchar>(i - 1, img.cols - 1);
	}
	for (int i = 1; i < img.cols + 1; ++i) {
		padding_img.at<uchar>(0, i) = img.at<uchar>(0, i - 1);
		padding_img.at<uchar>(img.rows + 1, i) = img.at<uchar>(img.rows - 1, i - 1);
	}
	return padding_img;
}


Mat Filtering2D(Mat img, Mat filter) {
	// 申请变量, 存储输出图像大小
	Mat filtered_img(img.rows - 2, img.cols - 2, CV_8UC1, Scalar(0));
	// img 转变为float 类型
	img.convertTo(img, CV_32F);

	for (int i = 0; i < filtered_img.rows; i++) {
		for (int j = 0; j < filtered_img.cols; j++) {
			// ###### 这里编程实现滤波公式 ##########
			float pixel = 0;
			for(int x = 0; x<filter.rows;x++){
				for(int y = 0; y< filter.cols;y++){
					pixel += img.at<float>(i+x,j+y) * filter.at<float>(x,y);
				}
			}

			// ############## 结束编程 #############
			filtered_img.at<uchar>(i, j) = saturate_cast<uchar>(pixel);
		}
	}

	return filtered_img;
}

cv::Mat denoisewithOrderStatisticsFilter(cv::Mat img) {
	cv::Mat filtered_img(img.rows - 2, img.cols - 2, CV_8UC1);

	for (int i = 0; i < filtered_img.rows; ++i) {
		for (int j = 0; j < filtered_img.cols; ++j) {
			// ###### 这里编程实现滤波公式 ##########
			float pixel = 0;
			int filter_size = 3;
			for (int x = 0; x < filter_size; x++) {
				for (int y = 0; y < filter_size; y++) {
					pixel += img.at<uchar>(i + x, j + y);
				}
			}
			pixel /= (filter_size * filter_size);
			// ############## 结束编程 #############
			filtered_img.at<uchar>(i, j) = static_cast<uchar>(pixel);
		}
	}

	return filtered_img;
}

cv::Mat denoisewithOrderStatisticsFilter_Mean(cv::Mat img) {
	cv::Mat filtered_img(img.rows - 2, img.cols - 2, CV_8UC1);

	for (int i = 0; i < filtered_img.rows; ++i) {
		for (int j = 0; j < filtered_img.cols; ++j) {
			// ###### 这里编程实现滤波公式 ##########
			float pixel = 0;
			int filter_size = 3;
			for (int x = 0; x < filter_size; x++) {
				for (int y = 0; y < filter_size; y++) {
					pixel += img.at<uchar>(i + x, j + y);
				}
			}
			pixel /= (filter_size * filter_size);
			// ############## 结束编程 #############
			filtered_img.at<uchar>(i, j) = static_cast<uchar>(pixel);
		}
	}
	return filtered_img;
}

cv::Mat denoisewithOrderStatisticsFilter_Median(cv::Mat img) {
	cv::Mat filtered_img(img.rows - 2, img.cols - 2, CV_8UC1);

	for (int i = 0; i < filtered_img.rows; ++i) {
		for (int j = 0; j < filtered_img.cols; ++j) {
			// ###### 这里编程实现滤波公式 ##########
			float pixel = 0;
			vector<uchar> neighbor; // 像素邻域
			int filter_size = 3;
			for (int x = 0; x < filter_size; x++) {
				for (int y = 0; y < filter_size; y++) {
					neighbor.push_back(img.at<uchar>(i + x, j + y));
				}
			}
			sort(neighbor.begin(), neighbor.end());
			pixel = neighbor[neighbor.size() / 2];
			// ############## 结束编程 #############
			filtered_img.at<uchar>(i, j) = static_cast<uchar>(pixel);
		}
	}

	return filtered_img;
}

cv::Mat denoisewithOrderStatisticsFilter_Max(cv::Mat img) {
	cv::Mat filtered_img(img.rows - 2, img.cols - 2, CV_8UC1);

	for (int i = 0; i < filtered_img.rows; ++i) {
		for (int j = 0; j < filtered_img.cols; ++j) {
			// ###### 这里编程实现滤波公式 ##########
			uchar pixel = 0;
			int filter_size = 3;
			for (int x = 0; x < filter_size; x++) {
				for (int y = 0; y < filter_size; y++) {
					pixel = max(pixel, img.at<uchar>(i + x, j + y));
				}
			}
			// ############## 结束编程 #############
			filtered_img.at<uchar>(i, j) = static_cast<uchar>(pixel);
		}
	}

	return filtered_img;
}

cv::Mat denoisewithOrderStatisticsFilter_Min(cv::Mat img) {
	cv::Mat filtered_img(img.rows - 2, img.cols - 2, CV_8UC1);

	for (int i = 0; i < filtered_img.rows; ++i) {
		for (int j = 0; j < filtered_img.cols; ++j) {
			// ###### 这里编程实现滤波公式 ##########
			uchar pixel = 255;  // 这里根据图片属性设置
			int filter_size = 3;
			for (int x = 0; x < filter_size; x++) {
				for (int y = 0; y < filter_size; y++) {
					pixel = min(pixel, img.at<uchar>(i + x, j + y));
				}
			}
			// ############## 结束编程 #############
			filtered_img.at<uchar>(i, j) = static_cast<uchar>(pixel);
		}
	}

	return filtered_img;
}

double getPSNR(Mat ori_img, Mat en_img)
{
	double MAX = 255;
	double total = 0;
	ori_img.convertTo(ori_img, CV_32F);
	en_img.convertTo(en_img, CV_32F);
	for (int i = 0; i < ori_img.rows; i++)
	{
		for (int j = 0; j < ori_img.cols; j++)
		{
			total += pow((ori_img.at<float>(i, j) - en_img.at<float>(i, j)), 2);
		}
	}
	double MSE = total / (ori_img.rows * ori_img.cols);
	double PSNR = 10 * log10(MAX * MAX / MSE);
	return PSNR;
}

int main()
{
	Mat img = imread("test/1_smooth.jpg");
	img = getGrayImg(img);
	imshow("orginal image", img);
	Mat img_padding = paddingWithNeighbor(img);
	Mat filtered_img = Filtering2D(img_padding, lapalicanKernel);
	imshow("filtered image", filtered_img);
	//imshow("Averagefiltered image", Filtering2D(img_padding, averageKernel));
	//imshow("weightAveragefiltered image", Filtering2D(img_padding, weightedAverageKernel));
	imshow("sobel_x_filtered image", Filtering2D(img_padding, sobel_x));
	imshow("sobel_x2_filtered image", Filtering2D(Filtering2D(img_padding, sobel_x),sobel_x));
	imshow("sobel_y2_filtered image", Filtering2D(Filtering2D(img_padding, sobel_y), sobel_y));
	imshow("sobel_y_filtered image", Filtering2D(img_padding, sobel_y));
	// 2. 将平滑后的图像行锐化高通滤波 查看结果



	//3. 利用均值、中值、最大值、最小值对椒盐、椒、盐噪声图像进行去噪声并查看结果
	//Mat img2 = imread("test/2.jpg");
	//img2 = getGrayImg(img2);
	//imshow("orginal image", img2);
	//Mat img_padding2 = paddingWithNeighbor(img2);
	//Mat filtered_img2 = denoisewithOrderStatisticsFilter_Mean(img_padding2);

	//imshow("Meanfiltered image", denoisewithOrderStatisticsFilter_Mean(img_padding2));
	//imshow("Medianfiltered image", denoisewithOrderStatisticsFilter_Median(img_padding2));
	//imshow("Maxfiltered image", denoisewithOrderStatisticsFilter_Max(img_padding2));
	//imshow("Minfiltered image", denoisewithOrderStatisticsFilter_Min(img_padding2));

	//imwrite("1_enhanced_sharppen.jpg", filtered_img2);


	waitKey(0);
	destroyAllWindows();
	cout << "PSNR = " << getPSNR(img, filtered_img) << endl;
}
///////////////////////////////////////////
//exp1.1
//////////////////////////////////////////

