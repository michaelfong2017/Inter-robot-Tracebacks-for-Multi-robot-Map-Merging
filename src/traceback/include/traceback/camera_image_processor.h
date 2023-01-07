#ifndef CAMERA_IMAGE_PROCESSOR_H_
#define CAMERA_IMAGE_PROCESSOR_H_

#include <vector>
#include <unordered_map>
#include <boost/thread.hpp>

#include <geometry_msgs/Pose.h>
#include <opencv2/core/utility.hpp>

namespace traceback
{
    struct PoseImagePair
    {
        geometry_msgs::Pose pose;
        cv::Mat image;
        ros::Time stamp;
        bool operator<(const PoseImagePair &rhs) const
        {
            return stamp < rhs.stamp;
        }
    };

    class CameraImageProcessor
    {
    public:
        friend class Traceback;

    private:
        std::unordered_map<std::string, cv::Mat> robots_to_current_image_;
        std::unordered_map<std::string, std::vector<PoseImagePair>> robots_to_all_images_;
    };
}
#endif