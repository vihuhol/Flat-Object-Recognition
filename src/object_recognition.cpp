
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

vector<KeyPoint> FindFeatureUseSURF(const Mat image)
{
	int minHessian = 400;

	SurfFeatureDetector detector( minHessian );

	vector<KeyPoint> keypoints_object;

	detector.detect( image, keypoints_object );

	return keypoints_object;
	
}
void Descriptor(const Mat image, const Mat test_image)
{

	imshow("image",image);
	std::cout<<"here!"<< std::endl;
	int minHessian = 400;

	SurfFeatureDetector detector( minHessian );

	vector<KeyPoint> keypoints_object, keypoint_test;

	detector.detect( image, keypoints_object );
	detector.detect( test_image, keypoint_test );

	SurfDescriptorExtractor extractor;

	Mat destriptor_object, destriptor_test;

	
	extractor.compute(image,  keypoints_object ,destriptor_object);

	extractor.compute(test_image, keypoint_test ,destriptor_test);



	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match( destriptor_object, destriptor_test,matches );


	double max_dist = 0; double min_dist = 100;

            //-- Вычисление максимального и минимального расстояния среди всех дескрипторов
                       // в пространстве признаков
	for( int i = 0; i < destriptor_object.rows; i++ )
	{ 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );

	//-- Отобрать только хорошие матчи, расстояние меньше чем 3 * min_dist
	vector< DMatch > good_matches;

	for( int i = 0; i < destriptor_object.rows; i++ )
	{ 
		if( matches[i].distance < 3 * min_dist )
		{ 
			good_matches.push_back( matches[i]); 
		}
	}  


	Mat img_matches;
    

    drawMatches(image,keypoints_object,test_image, keypoint_test, good_matches,img_matches);


	//imshow("image",img_matches);
	//waitKey();



	vector<Point2f> obj;
	vector<Point2f> scene;

	for( int i = 0; i < good_matches.size(); i++ )
	{
		obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
		scene.push_back(  keypoint_test [ good_matches[i].trainIdx ].pt ); 
	}

	Mat H = findHomography ( obj, scene, CV_RANSAC );
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image.cols, 0 );
	obj_corners[2] = cvPoint( image.cols, image.rows ); obj_corners[3] = cvPoint( 0, image.rows );
	std::vector<Point2f> scene_corners(4);

	//-- Отобразить углы целевого объекта, используя найденное преобразование, на сцену
	perspectiveTransform( obj_corners, scene_corners, H);

	//-- Соеденить отображенные углы
	line( img_matches, scene_corners[0] + Point2f( image.cols, 0), scene_corners[1] + Point2f( image.cols, 0), Scalar(0, 255, 0), 4 );
	line( img_matches, scene_corners[1] + Point2f( image.cols, 0), scene_corners[2] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[2] + Point2f( image.cols, 0), scene_corners[3] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[3] + Point2f( image.cols, 0), scene_corners[0] + Point2f( image.cols, 0), Scalar( 0, 255, 0), 4 );


	imshow("image",img_matches);
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

		image = imread(_path + image_file);

		test_image = imread(testImage,1);

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
