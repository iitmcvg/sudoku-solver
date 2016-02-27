#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cstdlib>
#include <iostream>
#include <cstdio>
using namespace std;
using namespace cv;
vector<vector<Point> > contours;
vector<vector<Point> > approx_poly;
vector<Vec4i> hierarchy;
vector<Mat> boxes;
const int BORDER_REMOVE_P = 0;

/*vector<Point> find_corners(vector<Point> contr, int thresh) {
	vector<Point>::iterator it;
	vector<Point> return_points;
	int dx,dy;
	for(it = contr.begin()+3 ; it != contr.end()-4 ; it++) {
		dx = abs((((it-3)->x) - (it->x)) - ((it->x) - (it+3)->x));
		dy = abs((((it-3)->y) - (it->y)) - ((it->y) - (it+3)->y));
		if((dx+dy) > thresh) {
			return_points.push_back(*it);
		}
	}
	return return_points;
}*/

int main (int argc, char *argv[]) {
	Mat box,src,img = imread(argv[1]),sudoku_box,harris,harris_norm,harris_scale;
	double contour_areas=0,temp;
	src = img;
	int count = 0,i,j,T,r,c,b,thresh=200;
	vector<Point> corners;

	//////////////////////////////////////// INITIALIZATION /////////////////////////////////
	cvtColor(img,img,COLOR_BGR2GRAY);
	sudoku_box = Mat::zeros( img.rows, img.cols, CV_32FC1);
	threshold(img,img,0,255,CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	harris = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	harris.setTo(Scalar(255));
	// ACCESSING EACH BOX IN THE SUDOKU

	/*c = img.cols/9; r = img.rows/9;
	for(i = 0 ; i < 9 ; i++) {
		for(j = 0 ; j < 9 ; j++) {
			Rect R(Point(c*i + (c*BORDER_REMOVE_P)/100 ,r*j + (r*BORDER_REMOVE_P)/100 ),Point(c*(i+1) - (c*BORDER_REMOVE_P)/100 , r*(j+1) - (r*BORDER_REMOVE_P)/100 ));
			box = img(R);
			boxes.push_back(box);
		}
	}*/

	///////////////////////////////////////////////////////////////////////////////////////

	/*namedWindow("display1",WINDOW_NORMAL);
	imshow("display1",(*(boxes.begin()+31)));
	waitKey(0);*/

	findContours( img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	int idx = 0;
	cout << contours.size() << endl;
	cout << hierarchy.size() << endl;
	for(i = 0, j = 0 ; i < contours.size() ; i++) {
		temp = contour_areas;
		contour_areas = max(contour_areas,(contourArea(contours[i],false)));
		if(contour_areas != temp) j = i;
	}
	drawContours( sudoku_box, contours, j, Scalar(255), 1, 8 );

 	/*sudoku_box.convertTo(harris,CV_32S);*/

	approx_poly.resize(1);
	cout << sudoku_box.checkVector(2) <<endl;    // sudoku_box.checkVector(2) should be positive. Only then it can be passed to approxPolyDP
	approxPolyDP( contours[j], approx_poly[0], 0.01*arcLength(contours[j], true), true);        // Mat sudoku_box ( source img), harris ( destination img)

	drawContours( sudoku_box, approx_poly, 0, Scalar(255), 1, 8 );

	/*cornerHarris( sudoku_box, harris, 2, 3, 0.04, BORDER_DEFAULT );
	normalize( harris, harris_norm, 0, 255, NORM_MINMAX, CV_8UC1, Mat() );
	convertScaleAbs( harris_norm, harris_scale);
	int flag=1;*/
	/*for( j = 0; j < harris.rows ; j++ )
	{
		for( i = 0; i < harris.cols; i++ )
		{
			if( (int) harris_norm.at<float>(j,i) > thresh )
			{
				circle( harris_scale, Point( i, j ), 5,  Scalar(255,0,0), 2, 8, 0 );
				cout << i << " " << j << endl;
				flag = 0;
				break;
			}
		}
		if(flag == 0) break;
	}*/
	//corners =  find_corners(contours[j],5);


	/*for( ; idx >= 0; idx = hierarchy[idx][0] )
	{
		drawContours( sudoku_box, contours, idx, Scalar(100), 2, 8, hierarchy );
	}*/

	/*for(i = 0 ; i < img.rows ; i++) {
		for(j = 0 ; j < img.cols ; j++) {
		b = img.at<Vec3b>(i,j)[0]; g = img.at<Vec3b>(i,j)[1]; r = img.at<Vec3b>(i,j)[2];
		img.at<Vec3b>(i,j)[0] = img.at<Vec3b>(i,j)[1] = img.at<Vec3b>(i,j)[2] = (r+g+b)/3;
		}
		}*/

	//resize(img,img,Size(img.cols/4,img.rows/4));

	//namedWindow("display1",WINDOW_NORMAL);
	namedWindow("display2",WINDOW_NORMAL);
	//resizeWindow("display",w,h);
	//imshow("display1",img);
	namedWindow("display1",WINDOW_NORMAL);
  namedWindow("test",WINDOW_NORMAL);
	imshow("test",harris);
	imshow("display1",sudoku_box);
	imshow("display2",src);
	waitKey(0);

	return 0;
}
