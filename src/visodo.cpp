/*

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "vo_features.h"

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#define CV3 1
#define USE_CUDA 0

using namespace cv;
using namespace std;


int main_test()
{
    //Create a random 3D scene
    cv::Mat points3D(1, 160, CV_64FC4);
    cv::randu(points3D, cv::Scalar(-5.0, -5.0, 1.0, 1.0), cv::Scalar(5.0, 5.0, 10.0, 1.0 ));


    //Compute 2 camera matrices
    cv::Matx34d C1 = cv::Matx34d::eye();
    cv::Matx34d C2 = cv::Matx34d::eye();
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);

    //C2(1, 3) = 2;
    C2(2, 3) = 2;

    //Compute points projection
    std::vector<cv::Vec2d> points1;
    std::vector<cv::Vec2d> points2;

    for(int i = 0; i < points3D.cols; i++)
    {
        cv::Vec3d hpt1 = C1*points3D.at<cv::Vec4d>(0, i);
        cv::Vec3d hpt2 = C2*points3D.at<cv::Vec4d>(0, i);

        hpt1 /= hpt1[2];
        hpt2 /= hpt2[2];

        cv::Vec2d p1(hpt1[0], hpt1[1]);
        cv::Vec2d p2(hpt2[0], hpt2[1]);

        points1.push_back(p1);
        points2.push_back(p2);

        std::cout << p1[0] << "\t" << p1[1] << "--"<< p2[0] << "\t" << p2[1] <<  std::endl;
    }

    //Print
    std::cout << C1 << std::endl;
    std::cout << C2 << std::endl;

    //Recover essential
    cv::Mat E = cv::findEssentialMat(points1, points2, K, cv::FM_RANSAC, 0.99, 1);
    cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 1, 0.9);

    std::cout << "E: " << E << std::endl;
    std::cout << "F: " << F << std::endl;

    cv::Mat R;
    cv::Mat t;
    cv::recoverPose(F, points1, points2, K, R, t);

    std::cout << "R: " << R << std::endl;
    std::cout << "t: " << t << std::endl;


    cv::Mat R5;
    cv::Mat t5;
    cv::recoverPose(E, points1, points2, K, R5, t5);

    std::cout << "R5: " << R5 << std::endl;
    std::cout << "t5: " << t5 << std::endl;
}

double dist_point(const cv::Point2f & point1,const cv::Point2f & point2) {
    return sqrt((point1.x - point2.x) * (point1.x - point2.x)
                +(point1.y - point2.y) * (point1.y - point2.y));
}

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& prev_points, vector<Point2f>& next_points, vector<uchar>& status)
{
    // this function automatically gets rid of points for which tracking fails

    vector<float> err;
    vector<Point2f> points1,points2,points0;
    vector<uchar> status_backtrack;
    Size winSize = Size(21, 21);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    points0 = prev_points;
    points1.resize(prev_points.size());
    points2.resize(prev_points.size());
    calcOpticalFlowPyrLK(img_1, img_2, points0, points1, status          , err, winSize, 3, termcrit, 0, 0.001);
    calcOpticalFlowPyrLK(img_2, img_1, points1, points2, status_backtrack, err, winSize, 3, termcrit, 0, 0.001);

    for (int i = 0; i < status.size(); i++) {
        status[i] = status[i] & status_backtrack[i];
    }

    prev_points.clear();
    next_points.clear();
    for (int i = 0; i < status.size(); i++) {
        if( status[i] && dist_point(points2[i],points0[i]) < 1.5 ) {
            prev_points.push_back(points0[i]);
            next_points.push_back(points1[i]);
        }
    }
    status.resize(next_points.size(),1);
}

void featureDetection(Mat img_1, vector<Point2f>& points1)
{
    // uses FAST as of now, modify parameters as necessary
    vector<KeyPoint> keypoints_1;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
    KeyPoint::convert(keypoints_1, points1, vector<int>());
}

using namespace cv;
using namespace std;

#define MIN_NUM_FEAT 2000
//#define PATH "D:/KITTI_VO/02/image_0/"
#define PATH "../KITTI_00/"
#define DATASET PATH"%06d.png"
double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)
{
    return 1;
    string line;
    ifstream myfile (PATH"/00.txt");
    int i = 0;
    double x = 0, y = 0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open()) {
        while ((getline(myfile, line)) && (i <= frame_id)) {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            // cout << line << '\n';
            for (int j = 0; j < 12; j++) {
                in >> z ;
                if (j == 7) y = z;
                if (j == 3)  x = z;
            }
            i++;
        }
        myfile.close();
    } else {
        cout << "Unable to open file";
        return 0;
    }

    return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;
}

std::vector<double> getTimes(const std::string & filename) {
    string line;
    ifstream myfile (filename);
    std::vector<double> times;
    if (myfile.is_open()) {
        while ((getline(myfile, line)) ) {
            std::istringstream in(line);
            // cout << line << '\n';
            double t ;
            in >> t ;
            times.push_back(t);
        }
        myfile.close();
    } else {
        cout << "Unable to open file";
        return times;
    }
    return times;
}

// #define DATASET "/home/james/image_processing/videostab/bags/c15-41b/%04d.png"
// Mat K = (Mat_<double>(3, 3) << 564.85980225, 0., 973.51554108, 0., 664.61627197, 424.90181213, 0, 0, 1);
// #define MAX_FRAME 171



//R0_rect:
Mat Rext = (Mat_<double>(3, 3) <<
            9.999239000000e-01 ,9.837760000000e-03,-7.445048000000e-03,
            -9.869795000000e-03,9.999421000000e-01,-4.278459000000e-03,
            7.402527000000e-03 ,4.351614000000e-03,9.999631000000e-01);
Mat trans = (Mat_<double>(3, 3) <<
             0 , -1 , 0,
             0 , 0  ,-1,
             1 , 0  , 0);

Mat K = (Mat_<double>(3, 3) <<
         718.8560,        0, 607.1928,
         0       , 718.8560, 185.2157,
         0       ,        0,       1);
#define MAX_FRAME 325
//P0: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00
//    0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00
//    0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
int main(int argc, char** argv)
{
    main_test();
    Mat img_1, img_2;
    Mat R_f, t_f; // the final rotation and tranlation vectors containing the

    ofstream myfile;
    myfile.open("results1_1.txt");

    double scale = 1.00;
    char filename1[200];
    char filename2[200];
    sprintf(filename1, DATASET, 0);
    sprintf(filename2, DATASET, 1);
    std::vector<double> times = getTimes(PATH"times.txt");

    char text[100];
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    // read the first two frames from the dataset
    Mat img_1_c = imread(filename1);
    Mat img_2_c = imread(filename2);

    if (!img_1_c.data || !img_2_c.data) {
        std::cout<< " --(!) Error reading images " << std::endl;
        return -1;
    }

    // we work with grayscale images
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

    // feature detection, tracking
    vector<Point2f> points1, points2;        // vectors to store the coordinates of the feature points
    featureDetection(img_1, points1);        // detect features in img_1
    vector<uchar> status;
    featureTracking(img_1, img_2, points1, points2, status); // track those features to img_2

    // recovering the pose and the essential matrix
    Mat E, R, t;
#if CV3
    E = findEssentialMat(points2, points1, K, RANSAC, 0.999, 1.0);
#else
    Mat F = findFundamentalMat(points2, points1, FM_RANSAC, 3., 0.99);
    E = K.t() * F * K;
#endif
    recoverPose(E, points2, points1, K, R, t);
    cout << R << endl;

    Mat prevImage = img_2;
    Mat currImage;
    vector<Point2f> prevFeatures = points2;
    vector<Point2f> currFeatures;

    char filename[100];

    R_f = R.clone();
    t_f = t.clone();

    clock_t begin = clock();

    // namedWindow("Road facing camera", WINDOW_AUTOSIZE);// Create a window for display.
    // namedWindow("Trajectory", WINDOW_AUTOSIZE);// Create a window for display.

    Mat traj = Mat::zeros(800,800, CV_8UC3);

    double bearing = 0;

#if USE_CUDA
    CudaFeatureTraker cuda_ft(MIN_NUM_FEAT,21,3,30);
#endif
    for (int numFrame = 2; numFrame < times.size() ; numFrame++) {
        sprintf(filename, DATASET, numFrame);

        double delta_t =times[numFrame] - times[numFrame-1];
        //cout << "delta_t = " << delta_t << endl;
        Mat currImage_c = imread(filename);

        if(currImage_c.empty()) {
            break;
        }
        cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
        vector<uchar> status;

#if USE_CUDA
        cuda_ft.cuda_featureDetectionAndTraking(prevImage, currImage,prevFeatures, currFeatures, status);
#else
        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
#endif

#if CV3
        E = findEssentialMat(currFeatures, prevFeatures, K, RANSAC, 0.999, 1.0);
#else
        Mat F = findFundamentalMat(currFeatures, prevFeatures, FM_RANSAC, 3., 0.99);
        E = K.t() * F * K;
#endif

        recoverPose(E, currFeatures, prevFeatures, K, R, t);
        //recoverPose(E, currFeatures, prevFeatures, R, t, K.at<double>(0,0), cv::Point(K.at<double>(0,2),K.at<double>(1,2)));
        cv::Mat euler;
        Rodrigues(R, euler);
        //R = trans*Rext*R;
        //cout << numFrame << " " << 180 * bearing / 3.14159 << endl;
        Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);

        // this (x, y) combination makes sense as observed from the source code of triangulatePoints on GitHub
        for (int i = 0; i < prevFeatures.size(); i++) {
            prevPts.at<double>(0, i) = prevFeatures.at(i).x;
            prevPts.at<double>(1, i) = prevFeatures.at(i).y;

            currPts.at<double>(0, i) = currFeatures.at(i).x;
            currPts.at<double>(1, i) = currFeatures.at(i).y;
        }

        scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));
        Mat tt = scale * (R * t) ;
        double distance = tt.at<double>(2)*tt.at<double>(2)+ tt.at<double>(0)*tt.at<double>(0) + tt.at<double>(1)*tt.at<double>(1);
        distance = sqrt(distance);
        //cout << "speed :" << t.at<double>(2) /scale << "\n";
        double speed =(3.6*distance/delta_t)  ;
#define TODEG (180.f/CV_PI)
        double yaw   = TODEG *euler.at<double>(1)/delta_t;
        double pitch = TODEG *euler.at<double>(0)/delta_t;
        double roll  = TODEG *euler.at<double>(2)/delta_t;

        cout << tt << endl;
        cout << "speed :"  << speed << " k/h\n"<<
             "yaw   = " << yaw    <<" deg/s\n" <<
             "pitch = " << pitch  <<" deg/s\n" <<
             "roll  = " << roll   <<" deg/s\n" <<
             "-------------------\n" ;


        if ((scale > 0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {
            t_f = t_f + scale * (R_f * t);
            R_f = R * R_f;
        } else {
            cout << "scale below 0.1, or incorrect translation" << endl;
        }

        // lines for printing results
        // myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;

#if !USE_CUDA
        // a redetection is triggered in case the number of features being tracked go below a particular threshold
        if (prevFeatures.size() < MIN_NUM_FEAT) {
            // cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
            // cout << "trigerring redection" << endl;
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        }
#endif

        prevImage = currImage.clone();

        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 150;
        circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 1);

        rectangle(traj, Point(10, 0), Point(1000, 80), CV_RGB(0, 0, 0), CV_FILLED);
        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm speed = %02fkmph",
                t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2),speed);
        putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

        for(int i = 0 ; i< currFeatures.size() ; i++) {
            cv::circle(currImage_c,currFeatures[i],2,CV_RGB(255,0,0));
            cv::line(currImage_c,currFeatures[i],prevFeatures[i],CV_RGB(0,0,255));

        }

        imshow("Road facing camera", currImage_c);
        imshow("Trajectory", traj);
        waitKey(10);
        prevFeatures = currFeatures;
    }

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Total time taken: " << elapsed_secs << "s" << endl;

// cout << R_f << endl;
// cout << t_f << endl;

    return 0;
}
