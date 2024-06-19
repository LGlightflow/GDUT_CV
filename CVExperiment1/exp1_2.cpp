
/*
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

Mat getGrayImg(Mat img)
{
	Mat gray(img.size(), CV_8UC1);
	Mat timg;
	img.convertTo(timg, CV_32FC3);
	for (int i = 0; i < timg.rows; i++)
	{
		for (int j = 0; j < timg.cols; j++)
		{
			// R*0.299 + G*0.587 + B*0.114
			float gray_intensity = timg.at<Vec3f>(i, j)[0] * 0.114 + timg.at<Vec3f>(i, j)[1] * 0.587 + timg.at<Vec3f>(i, j)[2] * 0.299;
			gray.at<uchar>(i, j) = round(gray_intensity);
		}
	}
	return gray;
}

// S累积分布
// Ps概率分布
// Pr直方图
void get_histogram(Mat gray_img, Mat& S, Mat& Ps)
{
	Mat Pr = Mat::zeros(256, 1, CV_32SC1);
	for (int i = 0; i < gray_img.rows; i++)
	{
		for (int j = 0; j < gray_img.cols; j++)
		{
			//1.这里编程实现直方图统计
			int pixelValue = static_cast<int>(gray_img.at<uchar>(i, j));
			Pr.at<int>(pixelValue)++;
			//结束编程
		}
	}

	//2.归一化直方图，获得概率分布技
	Ps = Mat::zeros(256, 1, CV_32FC1);
	for (int i = 0; i < 256; i++) {
		Ps.at<float>(i) = static_cast<float>(Pr.at<int>(i)) / (gray_img.rows * gray_img.cols);
	}
	//结束编程

	Mat cumsum = Mat::zeros(256, 1, CV_32FC1);
	float pre_sum = 0;
	for (int i = 0; i < 256; i++)
	{
		pre_sum += Ps.at<float>(i); //
		cumsum.at<float>(i) = pre_sum;
	}

	cumsum *= 255;
	S = Mat::zeros(256, 1, CV_8UC1);
	for (int i = 0; i < 256; i++)
	{
		//3.获得累计概率分布
		S.at<uchar>(i) = static_cast<uchar>(cumsum.at<float>(i));
		//结束编程
	}
}


Mat image_equalization(Mat img, Mat S)
{
	Mat img_eq = Mat::zeros(img.size(), CV_8UC1);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			//实现像素值的重映射
			img_eq.at<uchar>(i, j) = S.at<uchar>(img.at<uchar>(i, j));


			//结束编程
		}
	}
	return img_eq;
}

double getPSNR(Mat ori_img, Mat en_img)
{
	double MAX = 255.0;
	double total = 0.0;
	for (int i = 0; i < ori_img.rows; i++)
	{
		for (int j = 0; j < ori_img.cols; j++)
		{
			total += pow((ori_img.at<uchar>(i, j) - en_img.at<uchar>(i, j)), 2);
		}
	}
	double MSE = total / (ori_img.rows * ori_img.cols);
	double PSNR = 10 * log10(MAX * MAX / MSE);
	return PSNR;
}

int main()
{
	//改变不同文件，查看效果
	Mat img = imread("test/1_smooth.jpg");
	Mat gray = getGrayImg(img);
	Mat S, Ps;
	get_histogram(gray, S, Ps);

	for (int i = 0; i < S.rows; i++) {
		for (int j = 0; j < S.cols; j++) {
			Vec3b pixel = S.at<Vec3b>(i, j);
			std::cout << static_cast<float>(pixel[0]) << " ";
			std::cout << static_cast<float>(pixel[1]) << " ";
			std::cout << static_cast<float>(pixel[2]) << " ";
		}
		cout << endl;
	};

	Mat img_eq = image_equalization(gray, S);
	double psnr = getPSNR(gray, img_eq);
	cout << "psnr = " << psnr << endl;


	//visualization
	imshow("getGrayImg", gray);
	imshow("image_equalization", img_eq);
	imwrite("LenaRGBLow1_enhanced.jpg", img_eq);

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true, accumulate = false;
	Mat hist;
	calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	//白色是未归一化的直方图；红色是归一化后的直方图
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0)); //白
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 255, 255), 2, 8, 0);
	}

	Mat hist_eq;
	calcHist(&img_eq, 1, 0, Mat(), hist_eq, 1, &histSize, &histRange, uniform, accumulate);
	normalize(hist_eq, hist_eq, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist_eq.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(hist_eq.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("Histogram", histImage);
	waitKey(0);

}

*/


