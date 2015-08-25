
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2\nonfree\features2d.hpp"
#include "opencv2\calib3d\calib3d.hpp"

#include <fstream>
#include <iostream>

using namespace cv;



const char* params =
     "{ h | help          | false | print usage                                   }"
     "{   | sample-list   |       | path to list with image classes names         }"
     "{   | image         |       | image to detect objects on                    }"
     "{   | camera        | false | whether to detect on video stream from camera }";

void Descriptor(const Mat image, const Mat test_image)
{
	const int ransacThreshold = 3;
	imshow("image",image);

	SurfFeatureDetector detector;
	vector<KeyPoint> keypoints_object, keypoint_test;
	detector.detect( image, keypoints_object );
	detector.detect( test_image, keypoint_test );

	SurfDescriptorExtractor extractor;
	Mat destriptor_object, destriptor_test;	
	extractor.compute(image,  keypoints_object ,destriptor_object);
	extractor.compute(test_image, keypoint_test ,destriptor_test);

	BFMatcher matcher( NORM_L2 );
	vector< DMatch > matches;
	matcher.match( destriptor_object, destriptor_test,matches );


	Mat img_matches; 
    drawMatches(image,keypoints_object,test_image, keypoint_test, matches,img_matches);
	imshow("Matches before RANSAC",img_matches);
	waitKey();

	vector<Point2f> obj;
	vector<Point2f> scene;
	for( int i = 0; i < matches.size(); i++ )
	{
		obj.push_back( keypoints_object[i].pt );
		scene.push_back(  keypoint_test[ matches[i].trainIdx ].pt ); 
	}
	
	Mat H = findHomography ( Mat(obj), Mat(scene), CV_RANSAC, ransacThreshold);

	Mat  scene_corners;
	perspectiveTransform(Mat(obj), scene_corners, H);

	vector< DMatch > inliers;
	for (int i=0; i<matches.size(); i++)
	{
		Point2f p1 = keypoint_test.at( matches[i].trainIdx ).pt;
		Point2f p2 = scene_corners.at<Point2f>(matches[i].queryIdx);
		if ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) < 
			ransacThreshold * ransacThreshold)
		{
			inliers.push_back(matches[i]);
		}
	}

	Mat img_matches_a;
	drawMatches(image, keypoints_object,test_image, keypoint_test, inliers,img_matches_a);

	imshow("Matches after RANSAC",img_matches_a);
	waitKey();
}



int main(int argc, const char **argv)
{ 
	vector<SurfDescriptorExtractor> detector;
	Mat image, test_image;
    CommandLineParser parser(argc, argv, params);
    string sampleListFile = parser.get<string>("sample-list");


	string testImage = parser.get<string>("image");

	string _path = "c:/Users/ss2015/Documents/GitHub/Flat-Object-Recognition";

    std::ifstream sampleListFileReader(sampleListFile);
    char buff[50];
    while (sampleListFileReader.getline(buff, 50))
    {
        string str(buff);
        string image_file = str.substr(0,str.find(" "));


		std::cout<<std::endl;

	//	std::cout<<str<<std::endl;

		image = imread(_path + image_file,0);

		test_image = imread(testImage,0);

		std::cout<<_path + image_file<<std::endl;

		std::cout<<image.size()<<std::endl;

		std::cout<<test_image.size()<<std::endl;

		//imshow("image",image);
		//std::cout<<"here!"<< std::endl;

		Descriptor(image,test_image);

		break;


    }
    return 0;
}
