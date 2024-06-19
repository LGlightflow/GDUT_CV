//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//void RGB2YUV_enhance(Mat img, Mat& temp_YUV, Mat& res_rgb, float lightness_en = 3.5) {
//	temp_YUV = Mat::zeros(img.size(), CV_8UC3);
//	res_rgb = Mat::zeros(img.size(), CV_8UC3);
//	Mat timg;
//	img.convertTo(timg, CV_32FC3);
//
//	for (int i = 0; i < timg.rows; i++) {
//		for (int j = 0; j < timg.cols; j++) {
//
//			float B = timg.at<Vec3f>(i, j)[0];
//			float G = timg.at<Vec3f>(i, j)[1];
//			float R = timg.at<Vec3f>(i, j)[2];
//
//			float Y = 0.299f * R + 0.587f * G + 0.114f * B;
//			float U = 0.492f * (B - Y);
//			float V = 0.877f * (R - Y);
//
//
//			temp_YUV.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(Y);
//			temp_YUV.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(U + 128);
//			temp_YUV.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(V + 128);
//
//			Y *= lightness_en;
//			float tempR = Y + 1.140f * V;
//			float tempG = Y - 0.395f * U - 0.581f * V;
//			float tempB = Y + 2.032f * U;
//
//			res_rgb.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(tempB);
//			res_rgb.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(tempG);
//			res_rgb.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(tempR);
//		}
//	}
//
//
//}
//
//int main(int argc, char** argv) {
//	Mat img = imread("test/Lena.jpg");
//	if (img.empty()) {
//		cerr << "Image not found" << endl;
//		return -1;
//	}
//
//
//	Mat imgyuv, res_rgb;
//	RGB2YUV_enhance(img, imgyuv, res_rgb);
//	vector<Mat> channels;
//	split(imgyuv, channels);
//	imshow("orginal image", img);
//	imshow("Y", channels[0]);
//	imshow("U", channels[1]);
//	imshow("V", channels[2]);
//	imshow("Enhance Light", res_rgb);
//	waitKey(0);
//	destroyAllWindows();
//	return 0;
//}
