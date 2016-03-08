#include "opencv2/imgproc/imgproc.hpp"
#include <tesseract/baseapi.h>
#include "opencv2/highgui/highgui.hpp"
#include <cstdlib>
#include <iostream>
#include <cstdio>
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
vector<Mat> box(82);
const int BORDER_REMOVE_P = 0;


int FindIndex( Point2f mid_pt, int cols, int rows ) {
	int i,j,mc=9999,mr=9999,ind_x,ind_y;
	for(i = 1 ; i <= 9 ; i++) {
		if ( mid_pt.x - (2*i-1)*(cols/18) < mc ) {
			ind_x = i;
			mc = mid_pt.x - (2*i-1)*(cols/18);
		}
		if ( mid_pt.y - (2*i-1)*(rows/18) < mr ) {
			ind_y = i;
			mr = mid_pt.x - (2*i-1)*(rows/18);
		}
	}
	return (9*ind_x + ind_y);
}

bool comp(const pair<int, int>&i, const pair<int, int>&j) {
	return i.second + i.first < j.second + j.first;
}

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
	Mat src, img = imread(argv[1]);
	Mat persp_transf, sudoku_box,harris, harris_norm, harris_scale;
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
	harris = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	harris.setTo(Scalar(255));
	persp_transf = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	persp_transf.setTo(Scalar(255));
	//box = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	//box.setTo(Scalar(255));
	////////////////////////////////////////////////////////////////////////////////////////

	////////////////////////////// FINDING SUDOKU BOX /////////////////////////////////////
	findContours( img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	int idx = 0;
	//cout << contours.size() << endl;
	//cout << hierarchy.size() << endl;
	for(i = 0, j = 0 ; i < contours.size() ; i++) {
		temp = contour_areas;
		contour_areas = max(contour_areas,(contourArea(contours[i],false)));
		if(contour_areas != temp) j = i;
	}
	drawContours( sudoku_box, contours, j, Scalar(255), 2, 8 );


	approx_poly.resize(1);
	approxPolyDP( contours[j], approx_poly[0], 0.01*arcLength(contours[j], true), true);  // OBTAINING CORNER POINTS

	drawContours( sudoku_box, approx_poly, 0, Scalar(255), 2, 8 );    
	imshow("sudoku_box",sudoku_box);
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
	Mat src_gray, contours_img, num_extraction_img;
	warpPerspective( src, persp_transf, M, persp_transf.size(), INTER_LINEAR, BORDER_CONSTANT);
	Mat persp_transf_8UC1;
	cvtColor(persp_transf, persp_transf_8UC1, COLOR_BGR2GRAY);
	//threshold( persp_transf_8UC1, persp_transf_8UC1, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
	//imshow("tes", persp_transf_8UC1);

	GaussianBlur( persp_transf_8UC1, persp_transf_8UC1, Size(5,5), 0, 0, BORDER_DEFAULT );
	kernel = getStructuringElement( MORPH_RECT, Size(3,3) );
	//dilate( persp_transf_8UC1, persp_transf_8UC1, kernel, Point(-1,-1), 1 ); 
	//erode( persp_transf_8UC1, persp_transf_8UC1, kernel, Point(-1,-1), 1 ); 
	adaptiveThreshold( persp_transf_8UC1, persp_transf_8UC1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 0 );
	//threshold( persp_transf_8UC1, persp_transf_8UC1, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
	//imshow("tesr", persp_transf_8UC1);

	////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////// SEGMENTING SMALL BOXES ////////////////////////////////////////////
	int k,q,fl=0;
	Mat dummy2,persp_transf_8UC1_contours = Mat::zeros(img.rows, img.cols, CV_8UC1);
	dummy2 = persp_transf_8UC1.clone();
	num_extraction_img = persp_transf_8UC1.clone(); 

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

  /*Mat boxes[82];
	for(i = 0 ; i < 81 ; i++) {
		boxes[i] = Mat::zeros( img.rows/8, img.cols/8, CV_8UC1 );
		boxes[i].setTo(Scalar(255));
	}*/

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
	for(i = 0, box_index = 0 ; i < transf_contours.size() ; i = transf_hierarchy[i][0]) {
		temp = contour_areas;
		contour_areas = max(contour_areas,(contourArea(transf_contours[i],false)));
		if(contour_areas != temp) box_index = i;
	}
	for(i = transf_hierarchy[box_index][2] ; i >= 0 ; i = transf_hierarchy[i][0]) {
		if(contourArea(transf_contours[i],false) > 100) {
			drawContours( persp_transf_8UC1_contours, transf_contours, i, Scalar(255), 1, 8 );
			t++;

			approx_poly_boxes.resize(1);
			approxPolyDP( transf_contours[i], approx_poly_boxes[0], 0.01*arcLength(transf_contours[i], true), true);  // OBTAINING CORNER POINTS
			mid_pt = Point2f(0,0);

			for(it = approx_poly_boxes[0].begin() ; it != approx_poly_boxes[0].end() ; it++) { 
				mid_pt += Point2f( (it->x)/4, (it->y)/4 );
				box_vertices.push_back( make_pair(it->x, it->y) );
			}

			sort( box_vertices.begin(), box_vertices.end(), comp );
			itt = box_vertices.begin();
			if( (itt+1)->first < (itt+2)->first ) {
				tempr = *(itt+1);
				*(itt+1) = *(itt+2);
				*(itt+2) = tempr;
			}
			int index = FindIndex( mid_pt, img.cols, img.rows );
			i1 = 0;

			for(itt = box_vertices.begin() ; itt != box_vertices.end() ; itt++) {
			  small_boxes[index][i1] = Point2f( itt->first, itt->second );
				i1++;
			}
			box_vertices.erase( box_vertices.begin(), box_vertices.end() );
			approx_poly_boxes[0].erase( approx_poly_boxes[0].begin(), approx_poly_boxes[0].end() );
		}
	}
	//erode( persp_transf_8UC1, persp_transf_8UC1, kernel, Point(-1,-1), 1 ); 
	//dilate( persp_transf_8UC1, persp_transf_8UC1, kernel, Point(-1,-1), 1 ); 

	///////////////////////findContours( persp_transf_8UC1_contours, 

	cout << "t = " << t << endl;
	namedWindow("boxes_cont",WINDOW_NORMAL);
	resize( persp_transf_8UC1_contours, persp_transf_8UC1_contours, Size(img.cols, img.rows) );
	imshow("boxes_cont",persp_transf_8UC1_contours);
	//imshow( "TEST", boxes[q]);
	namedWindow("display2",WINDOW_NORMAL);
	imshow("display2",persp_transf_8UC1);

	Mat temp_box,dummy,dummy3;

	vector<Mat>::iterator mit;
	for(i = 0 ; i < 82 ; i++) {
		Mat temp_out(img.rows/8, img.cols/8, CV_8UC1, Scalar::all(255 ));
		temp_out.copyTo(box[i]);
		cout << box[i] << "--";
	}
	temp_box = Mat::zeros( img.rows/3, img.cols/3, CV_8UC1 );
	c = src.cols/9; r = src.rows/9;
	int iter,max;
	Point2f p1,p2;
	double areas;
	for(i = 0 ; i < 9 ; i++) {
		for(j = 0 ; j < 9 ; j++) {
			p1 = small_boxes[i*9+j][0]; 
			p2 = small_boxes[i*9+j][4]; 
			Rect R(p1,p2);
			dummy3 = Mat::zeros( img.rows/8, img.cols/8, CV_8UC1 );
			dummy3.setTo(Scalar(255));
			dummy3 = box[81].clone();
			dummy3 = persp_transf(R);
			cout << "in " << box.size();;
			cvtColor( dummy3, box[i*9+j], COLOR_BGR2GRAY );
			cout << "out ";
			threshold( box[i*9+j], box[i*9+j], 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
			//adaptiveThreshold( box[9*i+j], box[9*i+j], 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 3, 6 );
			//findContours( box[i*9+j], transf_contours, transf_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
			//cout << transf_contours.size() << " ";
			//drawContours( box[i*9+j], transf_contours, -1, Scalar(255), CV_FILLED, 8 );   
			/*if(9*i+j == q) {
				namedWindow("test",WINDOW_NORMAL);
				imshow("test",box[q]);
				waitKey(0);
				}*/



			/*for(k = 0 ;  ; k++) {
				if(box[i*9+j].at<uchar>(k,k) <= 200) {
					floodFill( box[i*9+j], Point(k,k), Scalar(255));
					fl = 1;
					break;
				}
			}*/

			/*int y,z,cnt_w=0,cnt_b=0;
			for(y = 0 ; y <= box[i*9+j].rows ; y++) {
				for(z = 0 ; z <= box[i*9+j].cols ; z++) {
					if(box[i*9+j].at<uchar>(z,y) >= 200) cnt_w++;
					else cnt_b++;
				}
			}*/
		//	cout << i*9+j << " : " <<  cnt_w << " " << cnt_b << endl;


			dummy = Mat::zeros( img.rows/8, img.cols/8, CV_8UC1 );
			dummy.setTo(Scalar(255));

			areas = 0;
			contour_areas = 0;
			cout << dummy.rows << " " << dummy.cols << " " << dummy.type() << " " << dummy.channels() << endl;
			findContours( dummy, transf_contours2, transf_hierarchy2, RETR_TREE, CHAIN_APPROX_SIMPLE );
			for(iter = 0, max = 0 ; iter < transf_contours2.size() ; iter++) {
				temp = areas;
				areas = ( areas > (contourArea(transf_contours2[iter],false)) ? areas : (contourArea(transf_contours2[iter],false)) );
				if(areas != temp) max = iter;
			}
			if(transf_contours2.size()) {
				Rect bound_rect = boundingRect( Mat(transf_contours2[max]) );
				temp_box = box[i*9+j](bound_rect);
				copyMakeBorder( temp_box, temp_box, 6, 6, 6, 6, BORDER_REPLICATE ); 
			}

			if(i*9+j == q) {
				namedWindow("GAF",WINDOW_NORMAL);
				imshow("GAF",temp_box);
				waitKey(0);
			}
			//cout << i*9+j << " : " << areas << endl;

			//resize( box[i*9+j], box[i*9+j], Size(img.cols/4,img.rows/4) );

			/*tesseract::TessBaseAPI tess;
				tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
				tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
				tess.SetImage((uchar*)box[q].data, box[q].cols, box[q].rows, 1, box[q].cols);

				char *out = tess.GetUTF8Text();
				cout << out << endl;*/

			//threshold( box[i*9+j], box[i*9+j], 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
			/*kernel = getStructuringElement( MORPH_RECT, Size(3,3) );
				erode( box[i*9+j], box[i*9+j], kernel, Point(-1,-1), 1 );*/ 
		}
	}

	tesseract::TessBaseAPI tess;
	tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
	tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
	tess.SetImage((uchar*)box[q].data, box[q].cols, box[q].rows, 1, box[q].cols);

	char *out = tess.GetUTF8Text();
	cout << out << endl;

	//namedWindow("display3",WINDOW_NORMAL);
	//resizeWindow("display",w,h);
	//imshow("display3",box);
	namedWindow("display1",WINDOW_NORMAL);
	namedWindow("test",WINDOW_NORMAL);
	imshow("test",box[q]);
	imshow("display1",sudoku_box);
	waitKey(0);

	return 0;
}
