
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

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
    ifstream sampleListFileReader(sampleListFile);
    char buff[50];
    sampleListFileReader >> buff;
    cout << buff;
}

int main(int argc, const char **argv)
{   
    SampleListParser(argc, argv);
    return 0;
}