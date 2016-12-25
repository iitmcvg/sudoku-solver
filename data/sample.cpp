#include "opencv2/imgproc/imgproc.hpp"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;
int main() {
	Mat img,src;
	img = imread("img12.jpg",0);
	imshow("af",img);
	waitKey(0);
	return 0;
}
	
