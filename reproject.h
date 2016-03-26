#include "opencv2/imgproc/imgproc.hpp"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
using namespace std;
using namespace cv;

#define CONST 5

Mat num[10];
char q[30];
void initialise() {
	for(int i = 1 ; i <= 9 ; i++) {
		sprintf(q,"./reqd_nos/%d.jpg",i);
			num[i] = imread(q);
		if(! num[i].data) {
			cout <<  "Could not open or find the image in reprojection part" << endl ;
			return;
		}
	}
	num[0] = Mat::zeros(200,200,CV_8UC3);
	num[0].setTo(Scalar(255,255,255));
}

void add_images( Mat& src1, Mat& src2, Mat& dst ) {
	int i,j,b,g,r;
	for(i = 0 ; i < min(src1.rows,src2.rows) ; i++) {
		for(j = 0 ; j < min(src1.cols,src2.cols) ; j++) {
			b = src1.at<Vec3b>(i,j)[0] + src2.at<Vec3b>(i,j)[0];
			g = src1.at<Vec3b>(i,j)[1] + src2.at<Vec3b>(i,j)[1];
			r = src1.at<Vec3b>(i,j)[2] + src2.at<Vec3b>(i,j)[2];
			dst.at<Vec3b>(i,j)[0] = (b > 255 ? 255 : b);
			dst.at<Vec3b>(i,j)[1] = (g > 255 ? 255 : g);
			dst.at<Vec3b>(i,j)[2] = (r > 255 ? 255 : r);
		}
	}
}

Scalar get_color( Mat& image, Mat& projection_img, Point2f p1, Point2f p2 ) {
	// 'image' is the transformed color image 
	Scalar color = Scalar( image.at<Vec3b>(p1.x+CONST,p1.y+CONST)[0], image.at<Vec3b>(p1.x+CONST,p1.y+CONST)[1], image.at<Vec3b>(p1.x+CONST,p1.y+CONST)[2]);
	return color;
}	

void paste_image( Mat& image, Mat& projection_img, int n, Point2f p1, Point2f p4 ) {
	Mat number_img = Mat::zeros( num[n].rows, num[n].cols, CV_8UC3 );
	number_img = num[n].clone();
	cvtColor( number_img, number_img, COLOR_BGR2GRAY );
	if(number_img.rows != 0 && number_img.cols != 0) {
		//cout << number_img.rows << " " <<  number_img.cols << endl;
		resize( number_img, number_img, Size(p4.x-p1.x, p4.y-p1.y) );
		threshold( number_img, number_img, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
		copyMakeBorder( number_img, number_img, p1.y, projection_img.rows-p4.y-1, p1.x, projection_img.cols-p4.x-1, BORDER_CONSTANT );
		cvtColor( number_img, number_img, COLOR_GRAY2BGR );
		//cout <<
		
		//cout << "46" << endl;
		add_images( projection_img, number_img, projection_img );
		//cout << "48" << endl;
	}
	else {
		cout << " ERROR IN REPROJECTION " << endl;
	}
	return;
}

//' Mat image ' has the dimensions of source image
void reproject( Mat& image, int soln[9][9], int num[9][9], Point2f corners[82][4] ) {
	initialise();
  Mat projection_img = Mat::zeros(image.rows, image.cols, CV_8UC3);
	//projection_img.setTo(Scalar(255,255,255));
	int i,j,k;
	char c[30];
	for(i = 0 ; i < 9 ; i++) {
		for(j = 0 ; j < 9 ; j++) {
			if(num[i][j] != 0) continue;
			//sprintf(c," ./reqd_nos/%d.jpg",soln[j][i]);
			paste_image( image, projection_img, soln[i][j], corners[i*9+j][0], corners[i*9+j][3]);
		}
	}
	//cout << "68" << endl;
	add_images( image, projection_img, image );
	//cout << "71" << endl;
}

void inverse_transform( Mat& inv_M, Mat& src, Mat& dst ) {
	warpPerspective( src, dst, inv_M, dst.size(), INTER_LINEAR, BORDER_CONSTANT);
}
