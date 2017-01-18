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

#define CV3 0
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
    F = K.t() * F * K;

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
    return 1;
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


std::vector<double> getScale() {
    std::vector<double> scales;
    string line;
    ifstream myfile (PATH"/00.txt");
    std::vector<double> times;
    int i = 0;
    double x = 0, y = 0, z = 0;
    double x_prev, y_prev, z_prev;
    while ((getline(myfile, line))) {
        z_prev = z;
        x_prev = x;
        y_prev = y;
        std::istringstream in(line);
        for (int j = 0; j < 12; j++) {
            in >> z ;
            if (j == 7) y = z;
            if (j == 3)  x = z;
        }
        i++;
        scales.push_back(sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)));
    }
    myfile.close();
    return scales;
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
int main(int argc, char** argv)
{
    //main_test(); return 1;
    char filename1[200];
    sprintf(filename1, DATASET, 0);
    std::vector<double> times = getTimes(PATH"times.txt");
    char text[100], filename[256];
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;

    Mat img_1,img_2;
    Mat img_1_c = imread(filename1);
    if (!img_1_c.data ) {
        std::cout<< " --(!) Error reading images " << std::endl;
        return -1;
    }
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    Mat prevImage = img_1;
    Mat currImage;
    vector<Point2f> prevFeatures;
    vector<Point2f> currFeatures;
    featureDetection(img_1, prevFeatures);// detect features in img_1
    Mat R_f = Mat::eye(3,3,CV_64F);
    Mat t_f = Mat::zeros(3,1,CV_64F);// the final rotation and tranlation vectors containing the
    Mat traj = Mat::zeros(800,800, CV_8UC3);

#if USE_CUDA
    CudaFeatureTraker cuda_ft(2*MIN_NUM_FEAT,21,3,30);
#endif
    std::vector<double> scales = getScale();
    for (int numFrame = 1; numFrame < times.size() ; numFrame++) {
        Mat E, R, t;
        sprintf(filename, DATASET, numFrame);
        double delta_t =times[numFrame] - times[numFrame-1];
        Mat currImage_c = imread(filename);
        if(currImage_c.empty())
            break; // error read image
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
        Mat tt = scales[numFrame] * (R * t) ;
        double distance = tt.at<double>(2)*tt.at<double>(2)+ tt.at<double>(0)*tt.at<double>(0) + tt.at<double>(1)*tt.at<double>(1);
        distance = sqrt(distance);
        double speed =(3.6*distance/delta_t)  ;
        cv::Mat euler;
        Rodrigues(R, euler);
#define TODEG (180.f/CV_PI)
        double pitch = TODEG *euler.at<double>(0)/delta_t;
        double yaw   = TODEG *euler.at<double>(1)/delta_t;
        double roll  = TODEG *euler.at<double>(2)/delta_t;
        if ((scales[numFrame] > 0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {
            t_f = t_f + scales[numFrame] * (R_f * t);
            R_f = R * R_f;
        } else {
            cout << "scale below 0.1, or incorrect translation" << endl;
        }
#if !USE_CUDA
        // a redetection is triggered in case the number of features being tracked go below a particular threshold
        if (prevFeatures.size() < MIN_NUM_FEAT) {
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        }
#endif

        // DRAWING ----------------------------------------
        prevImage = currImage.clone();
        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 150;
        circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 1);
        rectangle(traj, Point(0, 0), Point(1000, 100), CV_RGB(0, 0, 0), CV_FILLED);
        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm speed = %02fkmph",t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2),speed);
        putText(traj, text, cv::Point(10, 20), fontFace, fontScale, Scalar::all(255), thickness, 8);
        sprintf(text, "Yaw: %0.2f deg/s pitch = %0.2f deg/s roll = %0.2f deg/s",yaw,pitch,roll);
        putText(traj, text, cv::Point(10, 40), fontFace, fontScale, Scalar::all(255), thickness, 8);
        for(int i = 0 ; i< currFeatures.size() ; i++) {
            cv::circle(currImage_c,currFeatures[i],2,CV_RGB(255,0,0));
            cv::line(currImage_c,currFeatures[i],prevFeatures[i],CV_RGB(0,0,255));
        }

        imshow("Road facing camera", currImage_c);
        imshow("Trajectory", traj);
        if( waitKey(10) == 27)
            break;
        prevFeatures = currFeatures;
    }
    return 0;
}
