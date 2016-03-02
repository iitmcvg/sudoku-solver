#include "opencv2/imgproc/imgproc.hpp"
#include <tesseract/baseapi.h>
#include "opencv2/highgui/highgui.hpp"
#include <cstdlib>
#include <iostream>
#include <cstdio>
using namespace std;
using namespace cv;
vector<vector<Point> > contours, transf_contours;
vector<vector<Point> > approx_poly;
vector<Vec4i> hierarchy, transf_hierarchy;
Mat box[82];
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
	Mat src,img = imread(argv[1]),persp_transf,sudoku_box,harris,harris_norm,harris_scale;
	resize( img, img, Size(500,500) );
	Mat M,kernel;
	src = img.clone();
	const int w = img.cols, h = img.rows;
	double contour_areas=0,temp;
	int count = 0,i,j,T,r,c,b,thresh=200;
	vector<Point> corners;
	Point2f transf_pts[4],corner_pts[4];

	cvtColor( img, img, COLOR_BGR2GRAY );
	sudoku_box = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	GaussianBlur( img, img, Size(5,5), 0, 0, BORDER_DEFAULT );
	kernel = getStructuringElement( MORPH_RECT, Size(3,3) );
	erode( img, img, kernel, Point(-1,-1), 1 ); 
	//threshold( img, img, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
	adaptiveThreshold( img, img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 5, 0 );
	imshow("ASFD",img);
	waitKey(0);
	harris = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	harris.setTo(Scalar(255));
	persp_transf = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	persp_transf.setTo(Scalar(255));
	//box = Mat::zeros( img.rows, img.cols, CV_8UC1 );
	//box.setTo(Scalar(255));
	////////////////////////////////////////////////////////////////////////////////////////
	// ACCESSING EACH BOX IN THE SUDOKU

	/*c = img.cols/9; r = img.rows/9;
		for(i = 0 ; i < 9 ; i++) {
		for(j = 0 ; j < 9 ; j++) {
		Rect R(Point(c*i + (c*BORDER_REMOVE_P)/100 ,r*j + (r*BORDER_REMOVE_P)/100 ),Point(c*(i+1) - (c*BORDER_REMOVE_P)/100 , r*(j+1) - (r*BORDER_REMOVE_P)/100 ));
		box[i*9 + j] = img(R);
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
	imshow("ASFD",sudoku_box);
	waitKey(0);


	approx_poly.resize(1);
	approxPolyDP( contours[j], approx_poly[0], 0.01*arcLength(contours[j], true), true);  // OBTAINING CORNER POINTS

	drawContours( sudoku_box, approx_poly, 0, Scalar(255), 2, 8 );    
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
	Mat src_gray;
	warpPerspective( src, persp_transf, M, persp_transf.size(), INTER_LINEAR, BORDER_CONSTANT);
	Mat persp_transf_8UC1;
	cvtColor(persp_transf, persp_transf_8UC1, COLOR_BGR2GRAY);
	threshold( persp_transf_8UC1, persp_transf_8UC1, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU );
	imshow("test",persp_transf_8UC1);
	waitKey(0);

	////////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////// SEGMENTING SMALL BOXES ////////////////////////////////////////////
	int k,q,fl=0;
	Mat persp_transf_8UC1_contours = Mat::zeros(img.rows, img.cols, CV_8UC1);
	//findContours( persp_transf_8UC1, transf_contours, transf_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	/*drawContours( persp_transf_8UC1_contours, transf_contours, -1, Scalar(255), 2, 8 );
		namedWindow("display1",WINDOW_NORMAL);
		imshow("display1",persp_transf_8UC1_contours);
		waitKey(0);*/

	for(k = 0 ; k < persp_transf_8UC1.rows ; k++) {
		if(i*9+j == 5) cout << persp_transf_8UC1.at<uchar>(0,k) << " ";
		if(persp_transf_8UC1.at<uchar>(0,k) >= 200) {
			floodFill( persp_transf_8UC1, Point(0,k), Scalar(0));
			fl = 1;
			break;
		}
	}
	if(fl == 0) {
		for(k = 0 ; k < persp_transf_8UC1.cols ; k++) {
			if(persp_transf_8UC1.at<uchar>(k,0) >= 200) {
				floodFill( persp_transf_8UC1, Point(k,0), Scalar(0));
				break;
			}
		}
	}

	for(i = 0 ; i < 81 ; i++) {
		box[i] = Mat::zeros( img.rows/8, img.cols/8, CV_8UC1 );
		box[i].setTo(Scalar(255));
	}
	Mat temp_box,dummy;
	temp_box = Mat::zeros( img.rows/3, img.cols/3, CV_8UC1 );
	cin >> q;
	c = src.cols/9; r = src.rows/9;
	for(i = 0 ; i < 9 ; i++) {
		for(j = 0 ; j < 9 ; j++) {
			Rect R(Point(c*i, r*j),Point(c*(i+1), r*(j+1)));
			box[i*9+j] = persp_transf_8UC1(R);

			//adaptiveThreshold( box[9*i+j], box[9*i+j], 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 3, 6 );
			//findContours( box[i*9+j], transf_contours, transf_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
			//cout << transf_contours.size() << " ";
			//drawContours( box[i*9+j], transf_contours, -1, Scalar(255), CV_FILLED, 8 );   
			/*if(9*i+j == q) {
				namedWindow("test",WINDOW_NORMAL);
				imshow("test",box[q]);
				waitKey(0);
				}*/


			/*for(k = 0 ; k < box[i*9+j].rows ; k++) {
				if(i*9+j == 5) cout << box[i*9+j].at<uchar>(0,k) << " ";
				if(box[i*9+j].at<uchar>(0,k) >= 200) {
				floodFill( box[i*9+j], Point(0,k), Scalar(0));
				break;
				}
				}
				for(k = 0 ; k < box[i*9+j].cols ; k++) {
				if(box[i*9+j].at<uchar>(k,0) >= 200) {
				floodFill( box[i*9+j], Point(k,0), Scalar(0));
				break;
				}
				}
				for(k = 0 ; k < img.cols/9 ; k++) {
				if(box[i*9+j].at<uchar>(k,(img.rows/9)-1) == 255) {
				floodFill( box[i*9+j], Point(k,(img.rows/9)-1), Scalar(0));
				break;
				}
				}*/

			int iter,max;
			double areas;
			contour_areas = 0;
			dummy = box[i*9+j].clone();
			findContours( dummy, transf_contours, transf_hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE );
			for(iter = 0, max = 0 ; iter < transf_contours.size() ; iter++) {
				temp = areas;
				areas = ( areas > (contourArea(transf_contours[iter],false)) ? areas : (contourArea(transf_contours[iter],false)) );
				if(areas != temp) max = iter;
			}
			if(transf_contours.size()) {
				Rect bound_rect = boundingRect( Mat(transf_contours[max]) );
				temp_box = box[i*9+j](bound_rect);
				copyMakeBorder( temp_box, temp_box, 3, 3, 3, 3, BORDER_REPLICATE ); 
			}
			if(i*9+j == q) {
				namedWindow("GAF",WINDOW_NORMAL);
				imshow("GAF",temp_box);
				waitKey(0);
			}

			resize( box[i*9+j], box[i*9+j], Size(img.cols/4,img.rows/4) );
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
	namedWindow("display2",WINDOW_NORMAL);
	//resizeWindow("display",w,h);
	//imshow("display3",box);
	namedWindow("display1",WINDOW_NORMAL);
	namedWindow("test",WINDOW_NORMAL);
	imshow("test",box[q]);
	imshow("display1",sudoku_box);
	imshow("display2",src);
	waitKey(0);

	return 0;
}
