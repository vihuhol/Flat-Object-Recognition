
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2\nonfree\features2d.hpp"

#include <fstream>
#include <iostream>

using namespace cv;



const char* params =
     "{ h | help          | false | print usage                                   }"
     "{   | sample-list   |       | path to list with image classes names         }"
     "{   | image         |       | image to detect objects on                    }"
     "{   | camera        | false | whether to detect on video stream from camera }";


void Descriptor(const Mat image,vector<KeyPoint> keypoints_object)
{
	SurfDescriptorExtractor extractor;

	Mat destriptor_object;

	extractor.compute(image, keypoints_object,destriptor_object);


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
    CommandLineParser parser(argc, argv, params);
    string sampleListFile = parser.get<string>("sample-list");
    std::ifstream sampleListFileReader(sampleListFile);
    char buff[50];
    while (sampleListFileReader.getline(buff, 50))
    {
        string str(buff);
        string image_file = str.substr(0,str.find(" "));

    }
    return 0;
}
