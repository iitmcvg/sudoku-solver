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

#define RM_WHITE 3
#define THRESH_NUM 40
#define THRESH_NUM_DIFF 60
#define THRESH_DIST 

using namespace std;
using namespace cv;

vector<vector<Point> > contours, transf_contours2;
vector<vector<Point> > approx_poly;
vector<Vec4i> hierarchy, transf_hierarchy2;
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

void find_sudokuBox( Mat& img, Mat& dst, vector<vector<Point> >& cntrs, vector<vector<Point> >& apprx_poly, vector<Vec4i>& hier ) {
	int i,j;
	double temp,contour_areas = 0;
	findContours( img, cntrs, hier, RETR_TREE, CHAIN_APPROX_SIMPLE);
	int idx = 0;
	for(i = 0, j = 0 ; i < cntrs.size() ; i++) {
		temp = contour_areas;
		contour_areas = max(contour_areas,(contourArea(cntrs[i],false)));
		if(contour_areas != temp) j = i;
	}
	drawContours( dst, cntrs, j, Scalar(255), 2, 8 );

	apprx_poly.resize(1);
	approxPolyDP( cntrs[j], apprx_poly[0], 0.01*arcLength(contours[j], true), true);  // OBTAINING CORNER POINTS 
	sort_boundaryPoints(apprx_poly[0]);
	//sort (

	drawContours( dst, apprx_poly, 0, Scalar(255), 2, 8 );
	//imshow("box_contour", dst);
	//waitKey(0);
}

// Gets the perspective transformation of the color image 'src' and stores it in the referenced Mat object 'persp_transf' and if needed converts it into a binary image and stores the binary image in the referenced Mat object 'bin' 
void PerspTransform( vector<vector<Point> >& appx_poly, Point2f* transf_pts, Point2f* corner_pts, Mat& src, Mat& persp_transf, Mat& bin, bool binImg ) {
	vector<Point>::iterator it;
	int i = 0;
	// 'corner_pts' contains the coordinates of the corner points of the sudoku box in the source image
	for(it = appx_poly[0].begin() ; it != appx_poly[0].end() ; it++) { 
		corner_pts[i] = Point2f( it->x, it->y );
		i++;
	}
	// 'transf_pts' contains the new set of coordinates of transformed points
	transf_pts[0] = Point2f(0,0); 
	transf_pts[1] = Point2f(0,src.rows); 
	transf_pts[2] = Point2f(src.cols,src.rows); 
	transf_pts[3] = Point2f(src.cols,0); 

	Mat M = getPerspectiveTransform( corner_pts, transf_pts);
	warpPerspective( src, persp_transf, M, persp_transf.size(), INTER_LINEAR, BORDER_CONSTANT);
	if(binImg) {
		// 'bin' is the binary image of persp_transf
		cvtColor(persp_transf, bin, COLOR_BGR2GRAY);
		GaussianBlur( bin, bin, Size(5,5), 0, 0, BORDER_DEFAULT );
		adaptiveThreshold( bin, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 0 );

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
//	if (mc*mc + mr*mr > 
	return (9*ind_x + ind_y);
}

bool comp(const pair<int, int>&i, const pair<int, int>&j) {
	return (i.second + i.first) < (j.second + j.first);
}

bool indices_present[82] = {false};

void segmentBoxes( Mat& dummy2, Point2f small_boxes[81][4], int& boxes_detected, Mat& persp_transf_8UC1_contours ) { 
	int t = 0,box_index,i1,i2,i,count;
	double contour_areas,temp;
	Point2f mid_pt = Point2f(0,0);
	vector<vector<Point> > approx_poly_boxes,transf_contours;
	vector<Vec4i> transf_hierarchy;
	vector<pair<float, float> > box_vertices;
	pair<float, float> tempr;
	vector<pair<float, float> >::iterator itt;
	contour_areas = 0;

	findContours( dummy2, transf_contours, transf_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	//////////////////////////////// Finding contour with max. area, whose children will be the 81 boxes ////////////////////////
	for(i = 0, box_index = 0 ; i < transf_contours.size() ; i = transf_hierarchy[i][0]) {
		temp = contour_areas;
		contour_areas = max(contour_areas,(contourArea(transf_contours[i],false)));
		if(contour_areas != temp) box_index = i;
	}
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	vector<Point>::iterator it;
	count = 0;
	double small_row = 0, small_col = 0;
	for(i = transf_hierarchy[box_index][2] ; i >= 0 ; i = transf_hierarchy[i][0]) {
		if(contourArea(transf_contours[i],false) > 200) {
			drawContours( persp_transf_8UC1_contours, transf_contours, i, Scalar(255), 1, 8 );
			t++;

			approx_poly_boxes.resize(1);
			vector < vector<Point> > hull (1);;
			approxPolyDP( transf_contours[i], approx_poly_boxes[0], 0.05*arcLength(transf_contours[i], true), true);  // OBTAINING CORNER POINTS
			convexHull ( Mat(approx_poly_boxes[0]), hull[0], false ); 
			mid_pt = Point2f(0,0);

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
			small_col += (itt->first - (itt+1)->first);
			small_row += (itt->second - (itt+2)->second);
			int index = FindIndex( mid_pt, dummy2.cols, dummy2.rows );
			if (box_vertices.size() != 4) printf(" %lu %d \n", box_vertices.size(), index );
			indices_present[index] = true;
			i1 = 0;
			for(itt = box_vertices.begin() ; itt != box_vertices.end() ; itt++) {
				small_boxes[index][i1] = Point2f( itt->first, itt->second );
				i1++;
			}
			box_vertices.erase( box_vertices.begin(), box_vertices.end() );
			approx_poly_boxes[0].erase( approx_poly_boxes[0].begin(), approx_poly_boxes[0].end() );
		}
	}
	small_row /= (double)count;  small_col /= (double)count;
	/*
	for (int i = 0 ; i < 81 ; ++i) {
		if (!indices_present[i]) {
			int mpx = i/9, mpy = i%9;
			for (int j = 0 ; j < 4 ; ++j) {
				small_boxes[i][j] = Point2f (
				*/
	boxes_detected = t;
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

bool can_be_Number( Mat& img, int width, int height ) {
	int i, j, cnt = 0, b,g,r;
	double ratio, area = width*height, wh;
	for(i = RM_WHITE ; i < height - RM_WHITE ; i++) {
		for(j = RM_WHITE ; j < width - RM_WHITE ; j++) {
			r = abs(img.at<Vec3b>(j,i)[2] - img.at<Vec3b>(j,i-1)[2]); 
			g = abs(img.at<Vec3b>(j,i)[1] - img.at<Vec3b>(j,i-1)[1]); 
			b = abs(img.at<Vec3b>(j,i)[0] - img.at<Vec3b>(j,i-1)[0]); 
			if( r+g+b >= THRESH_NUM_DIFF ) cnt++;
		}
	}
	return ( cnt >= THRESH_NUM ); 
}

void detect_numbers( Mat& img, Point2f small_boxes[81][4], int out[9][9]) {  
	Mat temp_box, dummy3, temp_out;
	Mat kernel = getStructuringElement( MORPH_RECT, Size(3,3) );
	temp_box = Mat::zeros( img.rows/3, img.cols/3, CV_8UC1 );
	Point2f p1,p2,p3,p4;
	Rect bound_rect;
	int width, height, r, c, i, j;
	char *ch;

	dummy3 = Mat::zeros( img.rows, img.cols, CV_8UC1 );

	for(i = 0 ; i < 81 ; i++) out[i/9][i%9] = 0;

	vector<Mat>::iterator mit;
	for(i = 0 ; i < 82 ; i++) {
		Mat temp_out(img.rows, img.cols, CV_8UC1, Scalar::all(255));
		temp_out.copyTo(box[i]);
	}
	temp_out = Mat::zeros(img.cols, img.rows, CV_8UC1 );

	for(i = 0 ; i < 81 ; i++) {
		rectangle( temp_out, small_boxes[i][0], small_boxes[i][3], Scalar(255), 1, 8, 0);
	}

	c = img.cols; r = img.rows;
	for(i = 0 ; i < 9 ; i++) {
		for(j = 0 ; j < 9 ; j++) {
			p1 = small_boxes[i*9+j][0]; 
			p2 = small_boxes[i*9+j][1]; 
			p3 = small_boxes[i*9+j][2]; 
			p4 = small_boxes[i*9+j][3]; 
			width = p4.x - p1.x;
			height = p4.y - p1.y;
			c = img.cols; r = img.rows;
			Rect R( p1.x, p1.y, width, height );
			dummy3.setTo(Scalar(255));
			if(0 <= R.x && 0 <= R.width && R.x + R.width <= c && 0 <= R.y && 0 <= R.height && R.y + R.height <= r)
				dummy3 = img(R);
			else {
				bound_rect_error( R, c, r );
			}

			Mat gray_box = Mat::zeros( dummy3.size(), CV_8UC1 );
			cvtColor( dummy3, gray_box, COLOR_BGR2GRAY );
			//GaussianBlur( box[i*9+j], box[i*9+j], Size(7,7), 0, 0, BORDER_DEFAULT );
			threshold( gray_box, box[i*9+j], 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
			//adaptive_otsuThresholding( box[i*9+j], box[i*9+j], 0, 255 );
			erode( box[i*9+j], box[i*9+j], kernel, Point(-1,-1), 1 ); 

			bool number = can_be_Number( dummy3, width, height ); 

			if(number) {
				tesseract::TessBaseAPI tess;
				tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
				tess.SetVariable("tessedit_char_whitelist", "123456789");
				tess.SetPageSegMode( tesseract::PSM_SINGLE_CHAR );
				tess.SetImage((uchar*)gray_box.data, gray_box.size().width, gray_box.size().height, gray_box.channels(), gray_box.step1());
				ch = tess.GetUTF8Text();
				out[i][j] = atoi(ch);
				//cout << i << " " << j << ":  " << out[i][j] << endl;
			}
			else { 
				//cout << i << " " << j << endl;
				out[i][j] = 0;
			}
		}
	}
}

void new_idea ( Mat& image, Mat kernal ) {
	Scalar color = 255;
	floodFill( image, Point2f( 0, 0 ), color );
	dilate ( image, image, kernal, Point(-1,-1), 1);
	imshow ("he", image);
	waitKey(0);
}
/*
int find_number ( Mat& image ) {
	tesseract::TessBaseAPI tess;
	tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
	tess.SetVariable("tessedit_char_whitelist", "123456789");
	tess.SetPageSegMode( tesseract::PSM_SINGLE_CHAR );
	tess.SetImage((uchar*)gray_box.data, gray_box.size().width, gray_box.size().height, gray_box.channels(), gray_box.step1());
	ch = tess.GetUTF8Text();
	return atoi(ch);
}
*/
/*
#define ROWS 55
#define COLS 55
void put_numbers ( Mat& image ) {
	for (int i = 0 ; i < 81 ; ++i) {

	}
}
*/

void inv_binImg(Mat& img) {
  int i,j;
	for(i = 0 ; i < img.rows ; i++) {
		for(j = 0 ; j < img.cols ; j++) {
			if(img.at<uchar>(i,j) > 200) img.at<uchar>(i,j) = 0;
			else img.at<uchar>(i,j) = 255;
		}
	}
}

int main (int argc, char *argv[]) {

	//////////////////////////////////////// INITIALIZATION /////////////////////////////////
	Mat src, img = imread(argv[1]);
	if(! img.data) {
	  cout <<  "ERROR: Could not open or find the image" << endl ;
		return 0;
	}
	resize( img, img, Size(500,500) );
	imshow("original image",img);
	Mat persp_transf, sudoku_box;
	sudoku_box = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	Mat M,kernel;
	src = img.clone();
	const int w = img.cols, h = img.rows;
	double contour_areas=0,temp;
	int count = 0,i,j,T,r,b,c,thresh=200;
	vector<Point> corners;
	Point2f transf_pts[4],corner_pts[4];

	cvtColor( img, img, COLOR_BGR2GRAY );
	GaussianBlur( img, img, Size(5,5), 0, 0, BORDER_DEFAULT );
	adaptiveThreshold( img, img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2 );
	inv_binImg(img);

	persp_transf = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	persp_transf.setTo(Scalar(255));
	////////////////////////////////////////////////////////////////////////////////////////

	find_sudokuBox( img, sudoku_box, contours, approx_poly, hierarchy ); // Draws the detected sudoku box in the Mat sudoku_box  
	imshow ("img", sudoku_box);
	waitKey(0);
	if(approx_poly[0].size() != 4) {
		cout << "ERROR: Sudoku box not detected" << endl;
		return 0;
	}

	Mat persp_transf_8UC1;
	PerspTransform( approx_poly, transf_pts, corner_pts, src, persp_transf, persp_transf_8UC1, true ); 
	Mat inv_M =  getPerspectiveTransform( transf_pts, corner_pts);

	Mat dummy2, dummy4, dummy5, persp_transf_8UC1_contours = Mat::zeros(img.rows, img.cols, CV_8UC1);
	dummy2 = persp_transf_8UC1.clone();
	dummy5 = Mat::zeros(img.rows, img.cols, CV_8UC1);
	kernel = getStructuringElement( MORPH_RECT, Size(3,3) );
	dilate( dummy2, dummy2, kernel, Point(-1,-1), 1 ); 
	//erode( dummy2, dummy2, kernel, Point(-1,-1), 1 ); 
	Point2f small_boxes[81][4];
	int t=0;
	segmentBoxes( dummy2, small_boxes, t, persp_transf_8UC1_contours );  
	imshow ("transformed_image", persp_transf_8UC1_contours );
	waitKey(0);
	//dummy4 = persp_transf_8UC1_contours.clone();
	//new_idea (dummy4, kernel);
	//segmentBoxes( dummy4, small_boxes, t, dummy5 );  
	//imshow ("transformed_image2", dummy5 );
	//waitKey(0);
	Mat temp_out = Mat::zeros(img.cols, img.rows, CV_8UC1 );
	for(i = 0 ; i < 81 ; i++) {
		rectangle( temp_out, small_boxes[i][0], small_boxes[i][3], Scalar(255), 1, 8, 0);
	}
	//imshow ("boxes", temp_out);
	//waitKey(0);
	if( t != 81 ) {
  	cout << "ERROR: The number of boxes detected is " << t << ", which is not equal to 81" << endl;
		return 0;
	}

	int out[9][9];
	detect_numbers( persp_transf, small_boxes, out );  
	
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

	namedWindow("display3",WINDOW_NORMAL);
	imshow("display3",src);
	waitKey(0);
	return 0;
}
