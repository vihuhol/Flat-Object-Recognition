
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2\nonfree\features2d.hpp"

#include <fstream>
#include <iostream>

using namespace cv;



const char* params =
     "{ h | help          | false | print usage                                   }"
     "{   | sample-list   | false | path to list with image classes names         }"
     "{   | image         |       | image to detect objects on                    }"
     "{   | camera        | false | whether to detect on video stream from camera }";

vector<Mat> samples;
vector<string> classNames;

void SampleListParser(int argc, const char **argv)
{
    CommandLineParser parser(argc, argv, params);
    string sampleListFile = parser.get<string>("sample-list");
    std::ifstream sampleListFileReader(sampleListFile);
    char buff[50];
    sampleListFileReader >> buff;
    std::cout << buff;
}

void FindFeatureUseSURF(const Mat image)
{
	int minHessian = 400;

	SurfFeatureDetector detector( minHessian );

	vector<KeyPoint> keypoints_object;

	detector.detect( image, keypoints_object );
	
}

int main(int argc, const char **argv)
{   
    return 0;
}
