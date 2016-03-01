#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cstdlib>
#include <iostream>
#include <cstdio>
using namespace std;
using namespace cv;
vector<vector<Point> > contours, transf_contours;
vector<vector<Point> > approx_poly;
vector<Vec4i> hierarchy, transf_hierarchy;
Mat boxes[82];
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

	//////////////////////////////////////// INITIALIZATION /////////////////////////////////
	Mat box,src,img = imread(argv[1]),persp_transf,sudoku_box,harris,harris_norm,harris_scale;
	Mat M;
	src = img;
	const int w = img.cols, h = img.rows;
	double contour_areas=0,temp;
	int count = 0,i,j,T,r,c,b,thresh=200;
	vector<Point> corners;
	Point2f transf_pts[4],corner_pts[4];

	cvtColor(img,img,COLOR_BGR2GRAY);
	sudoku_box = Mat::zeros( img.rows, img.cols, CV_32FC1);
	threshold(img,img,0,255,CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
	harris = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	harris.setTo(Scalar(255));
	persp_transf = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	persp_transf.setTo(Scalar(255));
	box = Mat::zeros( img.rows, img.cols, CV_32SC1 );
	//box.setTo(Scalar(255));
  ////////////////////////////////////////////////////////////////////////////////////////
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
  
	////////////////////////////// FINDING SUDOKU BOX /////////////////////////////////////
	findContours( img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	int idx = 0;
	cout << contours.size() << endl;
	cout << hierarchy.size() << endl;
	for(i = 0, j = 0 ; i < contours.size() ; i++) {
		temp = contour_areas;
		contour_areas = max(contour_areas,(contourArea(contours[i],false)));
		if(contour_areas != temp) j = i;
	}
	drawContours( sudoku_box, contours, j, Scalar(255), 2, 8 );


	approx_poly.resize(1);
	approxPolyDP( contours[j], approx_poly[0], 0.01*arcLength(contours[j], true), true);  // OBTAINING CORNER POINTS

	drawContours( sudoku_box, approx_poly, 0, Scalar(255), 2, 8 );    
	////////////////////////////////////////////////////////////////////////////////////////////
	
	///////////////////////////// PERSPECTIVE TRANSFORMATION ///////////////////////////////////
	vector<Point>::iterator it;
	i = 0;
	for(it = approx_poly[0].begin() ; it != approx_poly[0].end() ; it++) { 
		cout << "(x,y) = " << "(" << it->x << " , " << it->y << ")" << endl;
		corner_pts[i] = Point2f( it->x, it->y );
		i++;
	}
	transf_pts[0] = Point2f(0,0); 
	transf_pts[1] = Point2f(0,img.rows); 
	transf_pts[2] = Point2f(img.cols,img.rows); 
	transf_pts[3] = Point2f(img.cols,0); 

	M = getPerspectiveTransform( corner_pts, transf_pts);
	warpPerspective( src, persp_transf, M, persp_transf.size(), INTER_LINEAR, BORDER_CONSTANT);
	threshold( persp_transf, persp_transf,0,255,CV_THRESH_BINARY_INV | CV_THRESH_OTSU );

	////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////// SEGMENTING SMALL BOXES ////////////////////////////////////////////
 /* Mat persp_transf_contours = Mat::zeros(img.rows, img.cols, CV_8UC1);
	findContours( persp_transf, transf_contours, transf_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	drawContours( persp_transf_contours, transf_contours, -1, Scalar(255), 2, 8 );
	namedWindow("display1",WINDOW_NORMAL);
	imshow("display1",persp_transf_contours);
	waitKey(0);*/

	count=0; r = 0;
	/*for(i = hierarchy[j][2] ; i >= 0 ; i = hierarchy[i][0]) {
		drawContours( box[r], contours, i, Scalar(255), 2, 8);
		r++;
		boxes.push_back(box);
		box.setTo(Scalar(0));
		cout << (count++) << " " ;
	}*/
	//vector<Mat>::iterator box_it;
	//for(box_it = boxes.begin() ; box_it != boxes.begin() + 11 ; box_it++) {
		//namedWindow("small_boxes",WINDOW_NORMAL);
		//imshow("small_boxes" , box);
		//waitKey(0);
	//}
	//cout << endl;
	//drawContours( box, contours, hierarchy[j][2], Scalar(255), 2, 8 );
	
	/*cornerHarris( sudoku_box, harris, 2, 3, 0.04, BORDER_DEFAULT );
	normalize( harris, harris_norm, 0, 255, NORM_MINMAX, CV_8UC1, Mat() );
	convertScaleAbs( harris_norm, harris_scale);
	int flag=1;
	for( j = 0; j < harris_norm.rows ; j++ )
	{
		for( i = 0; i < harris_norm.cols; i++ )
		{
			if( (int) harris_norm.at<float>(j,i) > thresh )
			{
				circle( harris_scale, Point( i, j ), 5,  Scalar(255,0,0), 2, 8, 0 );
				//cout << i << " " << j << endl;
				//flag = 0;
				//break;
			}
		}
		//if(flag == 0) break;
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

	//namedWindow("display3",WINDOW_NORMAL);
	namedWindow("display2",WINDOW_NORMAL);
	//resizeWindow("display",w,h);
	//imshow("display3",box);
	namedWindow("display1",WINDOW_NORMAL);
  namedWindow("test",WINDOW_NORMAL);
	imshow("test",persp_transf);
	imshow("display1",sudoku_box);
	imshow("display2",img);
	waitKey(0);

	return 0;
}
