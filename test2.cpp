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
using namespace std;
using namespace cv;
vector<vector<Point> > contours, transf_contours, transf_contours2;
vector<vector<Point> > approx_poly;
vector<Vec4i> hierarchy, transf_hierarchy, transf_hierarchy2;
int sudoku_num[9][9];
Mat box[82];
const int BORDER_REMOVE_P = 0;

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
	for(int u = 0 ; u < width ; u++) {
		if( b.at<uchar>(u, 0) >= 200 ) {
			floodFill( b, Point(u, 0), color);
		}
	}
	for(int u = 0 ; u < height ; u++) {
		if( b.at<uchar>(u, 0) >= 200 ) {
			floodFill( b, Point(0, u), color);
		}
	}
	for(int u = 0 ; u < width ; u++) {
		if( b.at<uchar>(u, width-1) >= 200 ) {
			floodFill( b, Point(u, height-1), color);
		}
	}
	for(int u = 0 ; u < height ; u++) {
		if( b.at<uchar>(u, 0) >= 200 ) {
			floodFill( b, Point(width-1, u), color);
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

int check[82];

int main (int argc, char *argv[]) {

	//////////////////////////////////////// INITIALIZATION /////////////////////////////////
	Mat src, img = imread(argv[1]);
	Mat persp_transf, sudoku_box;
	resize( img, img, Size(500,500) );
	cout << " img  dimensions : " << img.rows << " " << img.cols << " " << img.type() << " " << img.channels() << endl;
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
	//threshold( img, img, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
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

	drawContours( sudoku_box, approx_poly, 0, Scalar(255), 2, 8 );    
	//imshow("sudoku_box",sudoku_box);
	////////////////////////////////////////////////////////////////////////////////////////////

	///////////////////////////// PERSPECTIVE TRANSFORMATION ///////////////////////////////////
	vector<Point>::iterator it;
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
	//imshow("tes", persp_transf_8UC1);
	//imshow("tesr", persp_transf_8UC1);

	GaussianBlur( persp_transf_8UC1, persp_transf_8UC1, Size(5,5), 0, 0, BORDER_DEFAULT );
	kernel = getStructuringElement( MORPH_RECT, Size(3,3) );
	//dilate( persp_transf_8UC1, persp_transf_8UC1, kernel, Point(-1,-1), 1 ); 
	//erode( persp_transf_8UC1, persp_transf_8UC1, kernel, Point(-1,-1), 1 ); 
	adaptiveThreshold( persp_transf_8UC1, persp_transf_8UC1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 0 );
	//threshold( persp_transf_8UC1, persp_transf_8UC1, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );

	////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////// SEGMENTING SMALL BOXES ////////////////////////////////////////////
	int k,q,fl=0;
	Mat dummy2,persp_transf_8UC1_contours = Mat::zeros(img.rows, img.cols, CV_8UC1);
	dummy2 = persp_transf_8UC1.clone();
	//num_extraction_img = persp_transf_8UC1.clone(); 

	//resize( dummy2, dummy2, Size(2*dummy2.cols, 2*dummy2.rows) );
	dilate( dummy2, dummy2, kernel, Point(-1,-1), 1 ); 
	//erode( dummy2, dummy2, kernel, Point(-1,-1), 1 ); 
	findContours( dummy2, transf_contours, transf_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	//drawContours( persp_transf_8UC1_contours, transf_contours, -1, Scalar(100), 2, 8 );


	/*for(k = 0 ; k <= 30 ; k++) {
		if(persp_transf_8UC1.at<uchar>(k,persp_transf_8UC1.rows/3) >= 200) {
		floodFill( persp_transf_8UC1, Point(k,persp_transf_8UC1.rows/3), Scalar(0));
		fl = 1;
		}
		}*/

	cin >> q;

	int t = 0,box_index,i1,i2;
	Point2f mid_pt = Point2f(0,0), small_boxes[82][4];

	/*for(i = 0 ; i >= 0 ; i = transf_hierarchy[i][0]) {
		drawContours( persp_transf_8UC1_contours, transf_contours, i, Scalar(100), 2, 8 );
		drawContours( boxes[t], transf_contours, i, Scalar(0), 2, 8);
		t++;
		}*/
	vector<vector<Point> > approx_poly_boxes;
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
	for(i = transf_hierarchy[box_index][2] ; i >= 0 ; i = transf_hierarchy[i][0]) {
		if(contourArea(transf_contours[i],false) > 100) {
			drawContours( persp_transf_8UC1_contours, transf_contours, i, Scalar(255), 1, 8 );
			t++;

			approx_poly_boxes.resize(1);
			approxPolyDP( transf_contours[i], approx_poly_boxes[0], 0.04*arcLength(transf_contours[i], true), true);  // OBTAINING CORNER POINTS
			mid_pt = Point2f(0,0);

			for(it = approx_poly_boxes[0].begin() ; it != approx_poly_boxes[0].end() ; it++) { 
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
			i1 = 0;
			//cout << count << " " << approx_poly_boxes[0].size() << " :- " << endl; 
			for(itt = box_vertices.begin() ; itt != box_vertices.end() ; itt++) {
				small_boxes[index][i1] = Point2f( itt->first, itt->second );
				i1++;
				//cout << itt->first << " " << itt->second << endl;
			}
			//cout << " mp : " << mid_pt.x << " " << mid_pt.y << endl;
			//cout << index <<  endl;
			check[index]++;
			//cout << endl;
			box_vertices.erase( box_vertices.begin(), box_vertices.end() );
			approx_poly_boxes[0].erase( approx_poly_boxes[0].begin(), approx_poly_boxes[0].end() );
		}
	}
	//erode( persp_transf_8UC1, persp_transf_8UC1, kernel, Point(-1,-1), 1 ); 
	//dilate( persp_transf_8UC1, persp_transf_8UC1, kernel, Point(-1,-1), 1 ); 

	///////////////////////findContours( persp_transf_8UC1_contours, 

	cout << "t = " << t << endl;
	//namedWindow("boxes_cont",WINDOW_NORMAL);
	resize( persp_transf_8UC1_contours, persp_transf_8UC1_contours, Size(img.cols, img.rows) );
	//imshow("boxes_cont",persp_transf_8UC1_contours);
	//waitKey(0);

	//imshow( "TEST", boxes[q]);

	Mat temp_box, dummy, dummy3, temp_out;

	dummy3 = Mat::zeros( img.rows, img.cols, CV_8UC1 );

	vector<Mat>::iterator mit;
	for(i = 0 ; i < 82 ; i++) {
		Mat temp_out(img.rows, img.cols, CV_8UC1, Scalar::all(255));
		temp_out.copyTo(box[i]);
	}
	temp_out = Mat::zeros(img.cols, img.rows, CV_8UC1 );

	for(i = 0 ; i < 81 ; i++) {
		rectangle( temp_out, small_boxes[i][0], small_boxes[i][3], Scalar(255), 1, 8, 0);
	}
	//namedWindow("display2",WINDOW_NORMAL);
	//imshow("display2",temp_out);
	//waitKey(0);

	//fill_smallBoxes(

	temp_box = Mat::zeros( img.rows/3, img.cols/3, CV_8UC1 );
	int iter,max;
	Point2f p1,p2,p3,p4;
	double areas;
	int width, height, out[9][9];
	char *ch,ch1;

	for(i = 0 ; i < 81 ; i++) out[i/9][i%9] = 0;

	c = persp_transf.cols; r = persp_transf.rows;
	Rect bound_rect;

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
			if(0 <= R.x && 0 <= R.width && R.x + R.width <= c && 0 <= R.y && 0 <= R.height && R.y + R.height <= r)
				dummy3 = persp_transf(R);
			else {
				bound_rect_error( R, c, r );
			}

			cvtColor( dummy3, box[i*9+j], COLOR_BGR2GRAY );
			//GaussianBlur( box[i*9+j], box[i*9+j], Size(7,7), 0, 0, BORDER_DEFAULT );
			threshold( box[i*9+j], box[i*9+j], 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
			erode( box[i*9+j], box[i*9+j], kernel, Point(-1,-1), 1 ); 

			//adaptiveThreshold( box[9*i+j], box[9*i+j], 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 0 );

			remove_whiteBorders( box[i*9+j], width, height, Scalar(0) );
			//findContours( box[i*9+j], transf_contours, transf_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
			//drawContours( box[i*9+j], transf_contours, -1, Scalar(255), CV_FILLED, 8 );   

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
			else continue;

			threshold( temp_box, temp_box, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );

			if(i*9+j == q) {
				namedWindow("GAF",WINDOW_NORMAL);
				imshow("GAF",temp_box);
				waitKey(0);
			}
			//cout << i*9+j << " : " << areas << endl;

			//resize( box[i*9+j], box[i*9+j], Size(img.cols/4,img.rows/4) );

			tesseract::TessBaseAPI tess;
			tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
			tess.SetVariable("tessedit_char_whitelist", "0123456789");
			//PIX *pix = pixCreateHeader( temp_box.size().width, temp_box.size().height, temp_box.depth() );

			tess.SetPageSegMode( tesseract::PSM_SINGLE_CHAR );
			//tess.SetImage((uchar*)temp_box.data, box[i*9+j].cols, box[i*9+j].rows, 1, box[i*9+j].cols);
			tess.SetImage((uchar*)temp_box.data, temp_box.size().width, temp_box.size().height, temp_box.channels(), temp_box.step1());
			//tess.ProcessPage( pix, NULL, 0, &text );

			ch = tess.GetUTF8Text();
			out[i][j] = atoi(ch);
			//cout << *ch << "  ";
		}
		//cout << endl;
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

	//namedWindow("display3",WINDOW_NORMAL);
	//resizeWindow("display",w,h);
	imshow("display3",src);
	//namedWindow("display1",WINDOW_NORMAL);
	namedWindow("test",WINDOW_NORMAL);
	imshow("test",box[q]);
	//imshow("display1",sudoku_box);
	waitKey(0);

	return 0;
}
