#ifndef STD_LIBS
#define STD_LIBS
#include "opencv2/imgproc/imgproc.hpp"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include "opencv2/highgui/highgui.hpp"
#include "sudoku_solver.h"
#include "reproject.h"
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#endif

#define RM_WHITE 3
#define THRESH_CONT_AREA 0
#define THRESH_VALUE 100
#define THRESH_NUM_DIFF 3.00
#define AREA_THRESH_MIN 2200
#define AREA_THRESH_MAX 2700
#define THRESH_VAR 0
#define Square(a) ((a)*(a))

using namespace std;
using namespace cv;
vector<vector<Point> > contours, transf_contours, transf_contours2;
vector<vector<Point> > approx_poly;
vector<Vec4i> hierarchy, transf_hierarchy, transf_hierarchy2;
int sudoku_num[9][9];
Mat box[82];
const int BORDER_REMOVE_P = 0;

void swap_point(Point& p1, Point& p2) {
	Point temp;
	temp = p1;
	p1 = p2;
	p2 = temp;
}

void sort_boundaryPoints( vector<Point>& points ) {
	vector<Point>::iterator it,temp;
	int s = 1,x,y;
	while(s != 0) {
		s = 0;
		for(it = points.begin()+1 ; it != points.end() ; it++) {
			if((it-1)->x > it->x) {
				swap_point(*(it-1), *it); 
				s++;
			}
		}
	}
	if(points[0].y > points[1].y) {
		swap_point(points[0],points[1]); 
	}
	if(points[2].y < points[3].y) {
		swap_point(points[2],points[3]); 
	}
}

int FindIndex( Point2f mid_pt, int cols, int rows ) {
	int i,j,mc=9999,mr=9999,ind_x,ind_y;
	for(i = 1 ; i <= 9 ; i++) {
		if ( abs(mid_pt.x - (2*i-1)*(cols/18)) < mc ) {
			ind_x = i-1;
			mc = abs(mid_pt.x - (2*i-1)*(cols/18));
		}
		if ( abs(mid_pt.y - (2*i-1)*(rows/18)) < mr ) {
			ind_y = i-1;
			mr = abs(mid_pt.y - (2*i-1)*(rows/18));
		}
	}
	return (9*ind_x + ind_y);
}

bool comp(const pair<int, int>&i, const pair<int, int>&j) {
	return i.second + i.first < j.second + j.first;
}

void remove_whiteBorders( Mat& b, int width, int height, const Scalar color ) {
	int i,j,k;
	floodFill( b, Point2f( 0, 0 ), color );
	floodFill( b, Point2f( width-1, 0 ), color );
	floodFill( b, Point2f( 0, height-1 ), color );
	floodFill( b, Point2f(width-1, height-1), color );
	for(i = 0 ; i < RM_WHITE ; i++) {
		for(int u = 0 ; u < width ; u++) {
			if( b.at<uchar>(u, i) >= 200 ) {
				//floodFill( b, Point(u, i), color);
				b.at<Scalar>(u,i) = color;
			}
		}

		for(int u = 0 ; u < height ; u++) {
			if( b.at<uchar>(i, u) >= 200 ) {
				//floodFill( b, Point(i, u), color);
				b.at<Scalar>(i,u) = color;
			}
		}
		for(int u = 0 ; u < width ; u++) {
			if( b.at<uchar>(u, height-i-1) >= 200 ) {
				//floodFill( b, Point(u, height-i-1), color);
				b.at<Scalar>(u,height-i-1) = color;
			}
		}
		for(int u = 0 ; u < height ; u++) {
			if( b.at<uchar>(width-i-1, u) >= 200 ) {
				//floodFill( b, Point(width-i-1, u), color);
				b.at<Scalar>(width-i-1,u) = color;
			}
		}
	}
	return;
}

void bound_rect_error(Rect& R, int c, int r ) {
	cout << "ERROR" << endl;

	if(0 > R.x) {
		cout << " bounded_rect.x problem : " << R.x << endl;
	}
	if(0 > R.width) {
		cout << " bounded_rect.width problem : " << R.width << endl;
	}
	if(	R.x + R.width > c) {
		cout << " bounded_rect.x + bounded_rect.width problem : " << R.x << " " << R.width << endl;
	}
	if(0 > R.y) {
		cout << " bounded_rect.y problem : " << R.y << endl;
	}
	if(0 > R.height) {
		cout << " bounded_rect.height problem : " << R.height << endl;
	}
	if(	R.y + R.height > r) {
		cout << " bounded_rect.y + bounded_rect.height problem : " << R.y << " " << R.height << endl;

	}
}

void adaptive_otsuThresholding( Mat& src, Mat& dst, int l = 0, int u = 255 ) {
	int i,j,r,c;
	r = src.rows; 
	c = src.cols;
	Mat p1,p2,p3,p4;
	p1 = Mat::zeros( src.size(), CV_8UC1 );
	p2 = p1.clone() ; p3 = p1.clone() ; p4 = p1.clone() ;
	for(i = 0 ; i < r ; i++) {
		for(j = 0 ; j < c ; j++) {
			if(i <= r/2 && j <= c/2) {
				p1.at<uchar>(j,i) = src.at<uchar>(j,i);
			}
			else if(i > r/2 && j <= c/2) {
				p2.at<uchar>(j,i) = src.at<uchar>(j,i);
			}
			else if(i <= r/2 && j > c/2) {
				p3.at<uchar>(j,i) = src.at<uchar>(j,i);
			}
			else {
				p4.at<uchar>(j,i) = src.at<uchar>(j,i);
			}
		}
	}
	threshold( p1, p1, l, u, CV_THRESH_BINARY | CV_THRESH_OTSU );
	threshold( p2, p2, l, u, CV_THRESH_BINARY | CV_THRESH_OTSU );
	threshold( p3, p3, l, u, CV_THRESH_BINARY | CV_THRESH_OTSU );
	threshold( p4, p4, l, u, CV_THRESH_BINARY | CV_THRESH_OTSU );
	dst = Mat::zeros( src.size(), CV_8UC1 );
	add( p1, dst, dst );
	add( p2, dst, dst );
	add( p3, dst, dst );
	add( p4, dst, dst );
}

double get_varience ( int thresh, Mat& image ) {
	double var = 0, tot = image.rows*image.cols;
	for (int i = 0 ; i < image.rows ; ++i) {
		for (int j = 0 ; j < image.cols ; ++j) {
			var += Square(image.at<uchar>(j,i) - thresh);
		}
	}
	var /= tot;
	return var;
}
const int Thresh = 500;

int can_be_no2 ( Mat& image ) {    // binary image
	int i,j,k;
	vector<vector<Point> > cont;
	vector<Vec4i> hier;
	findContours( image, cont, hier, RETR_TREE, CHAIN_APPROX_SIMPLE);
	int idx = 0, temp = 0, contour_areas = 0;
	for(i = 0, j = 0 ; i < cont.size() ; i++) {
		temp = contour_areas;
		if (contour_areas < (contourArea(contours[i],false))) {
			j = i;
			contour_areas = (contourArea(contours[i],false));
		}
	}
	return contour_areas;
}

bool can_be_Number( Mat& img, int width, int height ) {
	int i, j, cnt = 0, b = 0;
	double ratio, area = width*height, wh;
	Mat edge = Mat::zeros (img.rows, img.cols, CV_8UC1);
	for(i = RM_WHITE ; i < height - RM_WHITE ; i++) {
		for(j = RM_WHITE ; j < width - RM_WHITE ; j++) {
			int dx = ( img.at<uchar>(i,j+1) - img.at<uchar>(i,j) );
			int dy = ( img.at<uchar>(i+1,j) - img.at<uchar>(i,j) );
			if( dx*dx + dy*dy > Thresh ) {
				edge.at<uchar>(i,j) = 0;
				++cnt;
			}
			else 
				edge.at<uchar>(i,j) = 255;
		}
		//if( abs(img.at<uchar>(j,i) - img.at<uchar>(j,i-1)) >= THRESH_NUM ) cnt++;
	}
	int num = (double)(THRESH_NUM_DIFF/100.00)*(area);
	return ( cnt >= num); 
}



int check[82] = {0};

int main (int argc, char *argv[]) {

	//////////////////////////////////////// INITIALIZATION /////////////////////////////////
	Mat src, img = imread(argv[1]);
	Mat persp_transf, sudoku_box;
	resize( img, img, Size(500,500) );
	imshow ("Original Image", img);
	//waitKey(0);
	cout << " Img  dimensions : " << img.rows << " " << img.cols << "\n Image type " << img.type() << "\nImage channel " << img.channels() << endl;
	Mat M,kernel;
	src = img.clone();
	const int w = img.cols, h = img.rows;
	double contour_areas=0,temp;
	int count = 0,i,j,T,r,b,c,thresh=200;
	vector<Point> corners;
	Point2f transf_pts[4],corner_pts[4];

	cvtColor( img, img, COLOR_BGR2GRAY );
	sudoku_box = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	GaussianBlur( img, img, Size(5,5), 0, 0, BORDER_DEFAULT );
	kernel = getStructuringElement( MORPH_RECT, Size(3,3) );

	//erode( img, img, kernel, Point(-1,-1), 1 ); 
	//dilate( img, img, kernel, Point(-1,-1), 1 );
	//threshold( persp_transf_8UC1, persp_transf_8UC1, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
	adaptiveThreshold( img, img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2 );
	for(i = 0 ; i < img.rows ; i++) {
		for(j = 0 ; j < img.cols ; j++) {
			if(img.at<uchar>(i,j) > 200) img.at<uchar>(i,j) = 0;
			else img.at<uchar>(i,j) = 255;
		}
	}
	persp_transf = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	persp_transf.setTo(Scalar(255));
	////////////////////////////////////////////////////////////////////////////////////////

	////////////////////////////// FINDING SUDOKU BOX /////////////////////////////////////
	findContours( img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	int idx = 0;
	for(i = 0, j = 0 ; i < contours.size() ; i++) {
		temp = contour_areas;
		contour_areas = max(contour_areas,(contourArea(contours[i],false)));
		if(contour_areas != temp) j = i;
	}
	drawContours( sudoku_box, contours, j, Scalar(255), 2, 8 );


	approx_poly.resize(1);
	approxPolyDP( contours[j], approx_poly[0], 0.01*arcLength(contours[j], true), true);  // OBTAINING CORNER POINTS

	vector<Point>::iterator it;

	sort_boundaryPoints(approx_poly[0]);

	drawContours( sudoku_box, approx_poly, 0, Scalar(255), 2, 8 );    
	//imshow("sudoku_box",sudoku_box);
	//waitKey(0);
	////////////////////////////////////////////////////////////////////////////////////////////

	///////////////////////////// PERSPECTIVE TRANSFORMATION ///////////////////////////////////
	i = 0;
	for(it = approx_poly[0].begin() ; it != approx_poly[0].end() ; it++) { 
		corner_pts[i] = Point2f( it->x, it->y );
		i++;
	}
	transf_pts[0] = Point2f(0,0); 
	transf_pts[1] = Point2f(0,img.rows); 
	transf_pts[2] = Point2f(img.cols,img.rows); 
	transf_pts[3] = Point2f(img.cols,0); 

	M = getPerspectiveTransform( corner_pts, transf_pts);
	Mat inv_M =  getPerspectiveTransform( transf_pts, corner_pts);
	Mat src_gray, contours_img, num_extraction_img;
	warpPerspective( src, persp_transf, M, persp_transf.size(), INTER_LINEAR, BORDER_CONSTANT);
	Mat persp_transf_8UC1;
	cvtColor(persp_transf, persp_transf_8UC1, COLOR_BGR2GRAY);
	//threshold( persp_transf_8UC1, persp_transf_8UC1, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
	//imshow("tesr", persp_transf_8UC1);

	GaussianBlur( persp_transf_8UC1, persp_transf_8UC1, Size(5,5), 0, 0, BORDER_DEFAULT );
	kernel = getStructuringElement( MORPH_RECT, Size(3,3) );
	//dilate( persp_transf_8UC1, persp_transf_8UC1, kernel, Point(-1,-1), 1 ); 
	//erode( persp_transf_8UC1, persp_transf_8UC1, kernel, Point(-1,-1), 1 ); 
	adaptiveThreshold( persp_transf_8UC1, persp_transf_8UC1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 0 );
	//threshold( persp_transf_8UC1, persp_transf_8UC1, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );

	////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////// SEGMENTING SMALL BOXES ////////////////////////////////////////////
	int k,q,fl=0;
	Mat dummy2,persp_transf_8UC1_contours = Mat::zeros(img.rows, img.cols, CV_8UC1);
	dummy2 = persp_transf_8UC1.clone();

	dilate( dummy2, dummy2, kernel, Point(-1,-1), 1 ); 
	//erode( dummy2, dummy2, kernel, Point(-1,-1), 1 ); 
	findContours( dummy2, transf_contours, transf_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	int t = 0,box_index,i1,i2;
	Point2f mid_pt = Point2f(0,0), small_boxes[82][4];

	vector<vector<Point> > approx_poly_boxes;
	approx_poly_boxes.resize(1);
	vector<pair<float, float> > box_vertices;
	pair<float, float> tempr;
	vector<pair<float, float> >::iterator itt;
	contour_areas = 0;

	//////////////////////////////// Finding contour with max. area, whose child will be the 81 boxes ////////////////////////
	for(i = 0, box_index = 0 ; i < transf_contours.size() ; i = transf_hierarchy[i][0]) {
		temp = contour_areas;
		contour_areas = max(contour_areas,(contourArea(transf_contours[i],false)));
		if(contour_areas != temp) box_index = i;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for(i = 0 ; i < 82 ; i++) check[i] = 0;

	count = 0;
	double area, avg_rows = 0, avg_cols = 0;
	for(i = transf_hierarchy[box_index][2] ; i >= 0 ; i = transf_hierarchy[i][0]) {
		if(contourArea(transf_contours[i],false) > 100) {
			drawContours( persp_transf_8UC1_contours, transf_contours, i, Scalar(255), 1, 8 );

			vector < vector<Point> > hull (1);
			approxPolyDP( transf_contours[i], approx_poly_boxes[0], 0.0475*arcLength(transf_contours[i], true), true);  // OBTAINING CORNER POINTS
			convexHull ( Mat(approx_poly_boxes[0]), hull[0], false );
			mid_pt = Point2f(0,0);

			area =  contourArea(hull[0]);
			if (area < AREA_THRESH_MIN || area > AREA_THRESH_MAX || hull[0].size() != 4) continue;
			avg_rows += sqrt(area);

			for(it = hull[0].begin() ; it != hull[0].end() ; it++) { 
				mid_pt += Point2f( (it->x), (it->y) );
				box_vertices.push_back( make_pair(it->x, it->y) );
			}
			mid_pt.x /= 4;
			mid_pt.y /= 4;

			sort( box_vertices.begin(), box_vertices.end(), comp );
			itt = box_vertices.begin();
			if( (itt+1)->first < (itt+2)->first ) {
				tempr = *(itt+1);
				*(itt+1) = *(itt+2);
				*(itt+2) = tempr;
			}
			count++;
			int index = FindIndex( mid_pt, img.cols, img.rows );
			check[index]++;
			 t++;
			i1 = 0;
			for(itt = box_vertices.begin() ; itt != box_vertices.end() ; itt++) {
				small_boxes[index][i1] = Point2f( itt->first, itt->second );
				i1++;
			}
			box_vertices.clear();
			approx_poly_boxes[0].clear();
		}
	}
	//erode( persp_transf_8UC1, persp_transf_8UC1, kernel, Point(-1,-1), 1 ); 
	//dilate( persp_transf_8UC1, persp_transf_8UC1, kernel, Point(-1,-1), 1 ); 
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	avg_rows /= count;
	int side = avg_rows/2;

	// cout << "Average sidelength of small boxes  = " << side << endl;

	double rw = img.rows, cl = img.cols;
	for (int i = 0 ; i < 81 ; ++i) {
		if (check[i]) continue;
		double x = (int)i/9, y = (int)i%9;
		double mid_x = (rw/18 + (rw/9)*x), mid_y = (cl/18 + (cl/9)*y);
		/********************* Predicting the position of the uncertain and undetected boxes ********************/
		if (i >= 9 && check[i-9] && i < 72 && check[i+9]) {
			mid_x = (small_boxes[i-9][1].x + small_boxes[i+9][0].x)/2; 
			mid_y = (small_boxes[i-9][1].y + small_boxes[i-9][3].y)/2; 
		}
		else if (i%9 >= 1 && check[i-1] && i%9 < 8 && check[i+1]) {
			mid_x = (small_boxes[i-1][3].x + small_boxes[i-1][2].x)/2; 
			mid_y = (small_boxes[i-1][3].y + small_boxes[i+1][1].y)/2; 
		}
		else if (i >= 9 && check[i-9] && i%9 != 8 && check[i+1]) {
			mid_x = (small_boxes[i+1][0].x + small_boxes[i+1][1].x)/2; 
			mid_y = (small_boxes[i-9][3].y + small_boxes[i-9][1].y)/2; 
		}
		else if (i < 72 && check[i+9] && i%9 != 8 && check[i+1]) {
			mid_x = (small_boxes[i+1][0].x + small_boxes[i+1][1].x)/2; 
			mid_y = (small_boxes[i+9][3].y + small_boxes[i+9][1].y)/2; 
		}
		else if (i >= 9 && check[i-9] && i%9 > 0 && check[i-1]) {
			mid_x = (small_boxes[i-1][3].x + small_boxes[i-1][2].x)/2; 
			mid_y = (small_boxes[i-9][3].y + small_boxes[i-9][1].y)/2; 
		}
		else if (i < 72 && check[i+9] && i%9 > 0 && check[i+1]) {
			mid_x = (small_boxes[i-1][3].x + small_boxes[i-1][2].x)/2; 
			mid_y = (small_boxes[i+9][0].y + small_boxes[i+9][2].y)/2; 
		}
		
		small_boxes[i][0] = Point2f ( mid_x - side, mid_y - side );
		small_boxes[i][3] = Point2f ( mid_x + side, mid_y + side );
	}


	// Contours of the interiors of the sudoku box
	resize( persp_transf_8UC1_contours, persp_transf_8UC1_contours, Size(img.cols, img.rows) );

	Mat temp_box, dummy, dummy3, temp_out;

	dummy3 = Mat::zeros( img.rows, img.cols, CV_8UC1 );

	vector<Mat>::iterator mit;
	for(i = 0 ; i < 82 ; i++) {
		Mat temp_out(img.rows, img.cols, CV_8UC1, Scalar::all(255));
		temp_out.copyTo(box[i]);
	}
	temp_out = Mat(img.cols, img.rows, CV_8UC1, Scalar::all(0) );


	for(i = 0 ; i < 81 ; i++) {
		rectangle( temp_out, small_boxes[i][0], small_boxes[i][3], Scalar(255), 1, 8, 0);
	}
	//namedWindow("display2",WINDOW_NORMAL);
	//imshow("display2",temp_out);
	//waitKey(0);


	temp_box = Mat::zeros( img.rows/3, img.cols/3, CV_8UC1 );
	int iter,max;
	Point2f p1,p2,p3,p4;
	double areas;
	int width, height, out[9][9];
	char *ch,ch1;

	for(i = 0 ; i < 81 ; i++) out[i/9][i%9] = 0;

	c = persp_transf.cols; r = persp_transf.rows;
	Rect bound_rect;

	int ct = 0;

	for(i = 0 ; i < 9 ; i++) {
		for(j = 0 ; j < 9 ; j++) {
			p1 = small_boxes[i*9+j][0]; 
			p2 = small_boxes[i*9+j][1]; 
			p3 = small_boxes[i*9+j][2]; 
			p4 = small_boxes[i*9+j][3]; 
			width = p4.x - p1.x;
			height = p4.y - p1.y;
			c = persp_transf.cols; r = persp_transf.rows;
			Rect R( p1.x, p1.y, width, height );
			dummy3.setTo(Scalar(255));
			if(0 <= R.x && 0 <= R.width && R.x + R.width < c && 0 <= R.y && 0 <= R.height && R.y + R.height < r)  /* ~ : <= to <*/
				dummy3 = persp_transf(R);
			else {
				bound_rect_error( R, c, r );
			}

			Mat gray_box = Mat::zeros( dummy3.size(), CV_8UC1 );
			cvtColor( dummy3, gray_box, COLOR_BGR2GRAY );
			//GaussianBlur( box[i*9+j], box[i*9+j], Size(7,7), 0, 0, BORDER_DEFAULT );
			int Thresh = threshold( gray_box, box[i*9+j], 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
			////cout << Thresh << endl;
			////double var = get_varience ( Thresh, gray_box );
			////var = sqrt(var);
			////adaptive_otsuThresholding( box[i*9+j], box[i*9+j], 0, 255 );
			erode( box[i*9+j], box[i*9+j], kernel, Point(-1,-1), 1 ); 

			bool number = can_be_Number( gray_box, width, height ); 

			//adaptiveThreshold( box[9*i+j], box[9*i+j], 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 0 );
			if(number) {

				dummy = Mat::zeros( img.rows/8, img.cols/8, CV_8UC1 );
				dummy.setTo(Scalar(255));

				dummy = box[i*9+j].clone();

				areas = 0;
				contour_areas = 0;
				findContours( dummy, transf_contours2, transf_hierarchy2, RETR_TREE, CHAIN_APPROX_SIMPLE );
				for(iter = 0, max = 0 ; iter < transf_contours2.size() ; iter++) {
					temp = areas;
					areas = ( areas > (contourArea(transf_contours2[iter],false)) ? areas : (contourArea(transf_contours2[iter],false)) );
					if(areas != temp) max = iter;
				}
				if (i*9 + j == 37) {
					imshow ("test",box[i*9+j]);
					waitKey(0);
				}

				if(transf_contours2.size()) {
					bound_rect = boundingRect( (transf_contours2[max]) );
					c = box[i*9+j].cols; r = box[i*9+j].rows;
					if(0 <= bound_rect.x && 0 <= bound_rect.width && bound_rect.x + bound_rect.width <= c && 0 <= bound_rect.y && 0 <= bound_rect.height && bound_rect.y + bound_rect.height <= r)
						temp_box = box[i*9+j](bound_rect);
					else 
						bound_rect_error( bound_rect, c, r );
					resize( temp_box, temp_box, Size(22,22) );
					copyMakeBorder( temp_box, temp_box, 4, 4, 4, 4, BORDER_CONSTANT ); 
				}
				else {
					continue;
				}

				/*temp_box = dummy3(bound_rect);
					cvtColor( temp_box, temp_box, COLOR_BGR2GRAY );
				 */

				threshold( temp_box, temp_box, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );

				if (i*9 + j == 37) {
					imshow ("temp_box",temp_box);
					waitKey(0);
				}

				/*resize( temp_box, temp_box, Size(22,22) );
					copyMakeBorder( temp_box, temp_box, 4, 4, 4, 4, BORDER_CONSTANT ); 
				 */
				/*
				if(i*9+j == q) {
					namedWindow("GAF",WINDOW_NORMAL);
					imshow("GAF",temp_box);
					waitKey(0);
				}
				*/

				//resize( box[i*9+j], box[i*9+j], Size(img.cols/4,img.rows/4) );

				tesseract::TessBaseAPI tess;
				tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
				tess.SetVariable("tessedit_char_whitelist", "123456789");
				//PIX *pix = pixCreateHeader( temp_box.size().width, temp_box.size().height, temp_box.depth() );

				tess.SetPageSegMode( tesseract::PSM_SINGLE_CHAR );
				//tess.SetImage((uchar*)temp_box.data, box[i*9+j].cols, box[i*9+j].rows, 1, box[i*9+j].cols);
				tess.SetImage((uchar*)temp_box.data, temp_box.size().width, temp_box.size().height, temp_box.channels(), temp_box.step1());
				//tess.ProcessPage( pix, NULL, 0, &text );

				ch = tess.GetUTF8Text();
				ct++;
				out[i][j] = atoi(ch);
			}
			else { 
				out[i][j] = 0;
			}
		}
	}
	
	int inp[9][9];
	for(i = 0 ; i < 9 ; i++) {
		for(j = 0 ; j < 9 ; j++) {
			inp[j][i] = out[j][i];
			cout << out[j][i] << "      ";
		}
		cout << endl;
	}

	if(solve(out)==1) {
		cout << "Solution" << endl;
		for(i = 0 ; i < 9 ; i++) {
			for(j = 0 ; j < 9 ; j++)
				cout << out[j][i] << "      ";
			cout << endl;
		}
	}
	else
		cout << "Not Possible" << endl;

	//presp_transf is the transformed color image
	Mat output_image, transformed_image = persp_transf.clone();
	Mat blank_image = Mat::zeros(img.rows, img.cols, CV_8UC3);
	output_image = Mat::zeros(img.rows, img.cols, CV_8UC3);
	reproject( blank_image, out, inp, small_boxes );
	inverse_transform( inv_M, blank_image, output_image ); 
  //copyMakeBorder( output_image, output_image, 50, 50, 50, 50, BORDER_CONSTANT );
	add_images( src, output_image, src); 

	imshow("display3",src);
	waitKey(0);

	return 0;
}
