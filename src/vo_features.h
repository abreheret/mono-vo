#pragma once

#ifndef __MONO_VO_H__
#define __MONO_VO_H__

#include "opencv2/core.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaimgproc.hpp"

class CudaFeatureTraker {

    cv::Ptr<cv::cuda::CornersDetector>        _detector;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> _pyrLK   ;
    int                                       _nb_point;

    static void _download(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec) {
        vec.resize(d_mat.cols);
        cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
        d_mat.download(mat);
    }
    static void _download(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec) {
        vec.resize(d_mat.cols);
        cv::Mat mat(1, d_mat.cols, CV_8U, (void*)&vec[0]);
        d_mat.download(mat);
    }

    static double _distance(const cv::Point2f & point1,const cv::Point2f & point2) {
        return sqrt((point1.x - point2.x) * (point1.x - point2.x)
                    +(point1.y - point2.y) * (point1.y - point2.y));
    }

public:
    CudaFeatureTraker(int nb_point,int winsize ,int max_level,int iters) {
        _detector = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, nb_point, 0.01);
        _pyrLK    = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(winsize, winsize), max_level, iters);
    }
    virtual ~CudaFeatureTraker() {}
    void cuda_featureDetectionAndTraking(const cv::Mat & img_1,
                                         const cv::Mat & img_2,
                                         std::vector<cv::Point2f> & prev_points,
                                         std::vector<cv::Point2f> & next_points,
                                         std::vector<uchar>& status) {
        cv::cuda::GpuMat frame0(img_1);
        cv::cuda::GpuMat frame1(img_2);
        cv::cuda::GpuMat prevPts;
        cv::cuda::GpuMat nextPts;
        cv::cuda::GpuMat gstatus;
        std::vector<cv::Point2f> points[3];

        cuda_featureDetection(frame0,prevPts);
        cuda_featureTracking(frame0,frame1,prevPts,nextPts,gstatus);

        cv::cuda::GpuMat gstatus_backtrack;
        cv::cuda::GpuMat points_bt;
        cuda_featureTracking(frame1,frame0,nextPts, points_bt, gstatus_backtrack);

        points[0].resize(prevPts.cols);
        _download(prevPts, points[0]);

        points[1].resize(nextPts.cols);
        _download(nextPts, points[1]);

        points[2].resize(points_bt.cols);
        _download(points_bt, points[2]);

        std::vector<uchar> status_backtrack;
        status.resize(gstatus.cols);
        _download(gstatus, status);
        status_backtrack.resize(gstatus_backtrack.cols);
        _download(gstatus_backtrack, status_backtrack);

        int indexCorrection = 0;

        for (int i = 0; i < status.size(); i++) {
            status[i] = status[i] & status_backtrack[i];
        }

        prev_points.clear();
        next_points.clear();
        for (int i = 0; i < status.size(); i++) {
            if( status[i] && _distance(points[2][i],points[0][i]) < 1.5 ) {
                prev_points.push_back(points[0][i]);
                next_points.push_back(points[1][i]);
            }
        }
        status.resize(next_points.size(),1);
    }

    void cuda_featureDetection(const cv::cuda::GpuMat & frame_gray, cv::cuda::GpuMat & pts) {
        _detector->detect(frame_gray, pts);
    }
    void cuda_featureTracking(const cv::cuda::GpuMat & img_1,
                              const cv::cuda::GpuMat & img_2,
                              cv::cuda::GpuMat & d_prevPts,
                              cv::cuda::GpuMat & d_nextPts,
                              cv::cuda::GpuMat & d_status) {
        _pyrLK->calc(img_1, img_2, d_prevPts, d_nextPts, d_status);
    }

};


#endif
