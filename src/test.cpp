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

cv::Mat rx(double theta) {
    cv::Mat R_x = (cv::Mat_<double>(3,3) <<
                   1, 0         , 0,
                   0, cos(theta), -sin(theta),
                   0, sin(theta), cos(theta));
    return R_x;
}

cv::Mat ry(double theta) {
    cv::Mat R_y = (cv::Mat_<double>(3,3) <<
                   cos(theta) ,    0, sin(theta),
                   0          ,    1, 0         ,
                   -sin(theta),    0, cos(theta));
    return R_y;
}
cv::Mat rz(double theta) {
    cv::Mat R_z = (cv::Mat_<double>(3,3) <<
                   cos(theta),-sin(theta), 0,
                   sin(theta), cos(theta), 0,
                   0         , 0         , 1);
    return R_z;
}

cv::Mat createMat3x4(const cv::Mat & m3x3,const cv::Mat & m3x1) {
    cv::Mat mat(3,4,m3x3.type());
    for (int i = 0 ; i < 3 ; i++)
        m3x3.col(i).copyTo(mat.col(i));
    m3x1.col(0).copyTo(mat.col(3));
    return mat;
}

cv::Mat createMat3x4(const cv::Mat & m3x3,const cv::Vec3d & v3) {
    return createMat3x4(m3x3,cv::Mat(v3));
}

cv::Mat eulerAnglesToRotationMatrix(cv::Vec3d &theta) { // Combined rotation matrix
    return rz(theta[2]) * ry(theta[1]) * rx(theta[0]);
}

int main_test(int nb_point,bool methode)
{
    if(methode == true && nb_point <= 5 || methode ==false && nb_point <= 8 ) {
        std::cout << "nb_point is too weak" << std::endl;
        return 1;
    }

    //Create a random 3D scene
    cv::Mat points3D(nb_point,1, CV_64FC4);
    cv::randu(points3D, cv::Scalar(-5.0, -5.0, 3.0, 1.0), cv::Scalar(5.0, 5.0, 10.0, 1.0 ));

    //Compute 2 camera matrices
    cv::Mat_<double> K(3, 3, CV_64F);
    K <<
      1*700,0,100,
      0  ,700,100,
      0  ,0,1;
    cv::Mat R1 = eulerAnglesToRotationMatrix(cv::Vec3d(0,0.,0));
    cv::Mat R2 = eulerAnglesToRotationMatrix(cv::Vec3d(0,-CV_PI/4,CV_PI/6));
    cv::Vec3f T1 = cv::Vec3f(0,0,0);
    cv::Vec3f T2 = cv::Vec3f(2,1,2);
    cv::Mat RT1 = createMat3x4(R1,T1);
    cv::Mat RT2 = createMat3x4(R2,T2);
    cv::Mat Proj1 = K*RT1;
    cv::Mat Proj2 = K*RT2;

    //Compute points projection
    std::vector<cv::Vec2d> points1;
    std::vector<cv::Vec2d> points2;
    printf(" X \t Y \t Z \t| \t u1 \t v1 \t u2 \t v2\n");
    for(int i = 0; i < points3D.rows; i++) {
        cv::Vec4d in = points3D.at<cv::Vec4d>(i);
        cv::Mat point3d(4,1, CV_64F);
        point3d.at<double>(0) = in [0];
        point3d.at<double>(1) = in [1];
        point3d.at<double>(2) = in [2];
        point3d.at<double>(3) = in [3];

        cv::Mat hpt1 = Proj1*point3d;
        cv::Mat hpt2 = Proj2*point3d;

        hpt1 /= hpt1.at<double>(2);
        hpt2 /= hpt2.at<double>(2);

        cv::Vec4d p3d = point3d;
        points1.push_back(cv::Vec2d(hpt1.at<double>(0),hpt1.at<double>(1)));
        points2.push_back(cv::Vec2d(hpt2.at<double>(0),hpt2.at<double>(1)));
        printf("%0.3f\t%0.3f\t%0.3f\t| \t%0.3f\t%0.3f\t%0.3f\t%0.3f\n",
               p3d[0],p3d[1],p3d[2],
               points1[i][0],points1[i][1],points2[i][0],points2[i][1]);
    }
    //Recover essential
    cv::Mat E;
    if(methode) {
        E = cv::findEssentialMat  (points1, points2, K, cv::FM_RANSAC, 0.99, 1);
    } else {
        cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3, 0.9);
        E = K.t() * F * K;
    }

    cv::Mat Rf;
    cv::Mat tf;
    cv::recoverPose(E, points1, points2, K, Rf, tf);
    cv::Mat RTf = createMat3x4(Rf,tf);
    cv::Mat points_trianguled;
    cv::triangulatePoints(Proj1,K*RTf,points1,points2,points_trianguled);
    for(int i = 0; i < points_trianguled.cols; i++) {
        points_trianguled.col(i) = points_trianguled.col(i) / points_trianguled.at<double>(3,i) ;
    }
    double scale = points3D.at<cv::Vec4d>(0)[2] / (points_trianguled.at<double>(2,0));
    double eps = 0.00001;
    std::cout << std::endl;
    cv::Mat Rtfs =createMat3x4(Rf,scale*tf);
    std::cout << "RT2 = \n"<< RT2  << std::endl << std::endl ;
    std::cout << "RTfs = \n"<< Rtfs << std::endl << std::endl;
    cv::Mat diff = RT2-Rtfs;
    double * ptr = (double*)diff.data;
    for(int i = 0 ; i < diff.rows*diff.cols ; i++) {
        ptr[i] = abs(ptr[i]) < eps ? 0 : ptr[i] ;
    }
    std::cout << "RT2-Rtfs = \n"<< diff << std::endl << std::endl;

    printf(" X \t Y \t Z \t\t X \t Y \t Z \n");
    for(int i = 0; i < points_trianguled.cols; i++) {
        cv::Vec4d p3d_tria = points_trianguled.col(i);
        cv::Vec4d p3d_real = points3D.at<cv::Vec4d>(i);
        p3d_tria = scale * p3d_tria;
        printf("%0.3f\t%0.3f\t%0.3f\t\t%0.3f\t%0.3f\t%0.3f",p3d_tria[0],p3d_tria[1],p3d_tria[2],p3d_real[0],p3d_real[1],p3d_real[2]);
        if (    abs(p3d_tria[0]-p3d_real[0]) < eps &&
                abs(p3d_tria[1]-p3d_real[1]) < eps &&
                abs(p3d_tria[2]-p3d_real[2]) < eps )
            printf("\tOK\n");
        else
            printf("\tKO\n");
    }
    return 1;
}