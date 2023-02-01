#ifndef CAMERA_IMAGE_PROCESSOR_H_
#define CAMERA_IMAGE_PROCESSOR_H_

#include <traceback/estimation_internal.h>

#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <boost/thread.hpp>

#include <geometry_msgs/Pose.h>
#include <sensor_msgs/Image.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>

namespace traceback
{
    struct PoseImagePair
    {
        geometry_msgs::Pose pose;
        sensor_msgs::Image image;
        int64_t stamp;
        bool operator<(const PoseImagePair &rhs) const
        {
            return stamp < rhs.stamp;
        }
    };

    // In robot world coordinates
    // tx and ty are in meters, r is in radian
    // Unit length
    struct TransformNeeded
    {
        double tx;
        double ty;
        double r;
    };

    class CameraImageProcessor
    {
    public:
        friend class Traceback;

        /*
        Return whether traced image matches tracer image, depending on the confidence.

        If match, output to transform_needed.

        header:
        seq: 15307
        stamp:
            secs: 1089
            nsecs: 114000000
        frame_id: "tb3_1/camera_rgb_optical_frame"
        height: 480
        width: 640
        distortion_model: "plumb_bob"
        D: [0.0, 0.0, 0.0, 0.0, 0.0]
        K: [530.4669406576809, 0.0, 320.5, 0.0, 530.4669406576809, 240.5, 0.0, 0.0, 1.0]
        R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        P: [530.4669406576809, 0.0, 320.5, -37.13268584603767, 0.0, 530.4669406576809, 240.5, 0.0, 0.0, 0.0, 1.0, 0.0]
        binning_x: 0
        binning_y: 0
        roi:
        x_offset: 0
        y_offset: 0
        height: 0
        width: 0
        do_rectify: False
        */
        bool findFurtherTransformNeeded(const cv::Mat &tracer_robot_image, const cv::Mat &traced_robot_image, FeatureType feature_type,
                                        double confidence, double yaw, TransformNeeded &transform_needed, std::string tracer_robot = "", std::string traced_robot = "", std::string current_time = "");

    private:
        std::unordered_map<std::string, sensor_msgs::Image> robots_to_current_image_;
        std::unordered_map<std::string, geometry_msgs::Pose> robots_to_current_pose_;
        std::unordered_map<std::string, std::list<PoseImagePair>> robots_to_all_pose_image_pairs_;
        std::unordered_map<std::string, std::unordered_set<size_t>> robots_to_all_visited_pose_image_pair_indexes_;

        cv::Vec3d rotationMatrixToEulerAngles(cv::Mat &R);
    };
}
#endif