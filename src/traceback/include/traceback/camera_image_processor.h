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
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>

#include <pcl/conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace traceback
{
    struct PoseImagePair
    {
        geometry_msgs::Pose pose;
        sensor_msgs::Image image;
        sensor_msgs::PointCloud2 point_cloud;
        int64_t stamp;
        bool operator<(const PoseImagePair &rhs) const
        {
            return stamp < rhs.stamp;
        }
    };

    // In robot world coordinates
    // tx and ty are in meters, r is in radian
    // Any length
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

        bool pointCloudMatching(const sensor_msgs::PointCloud2 &tracer_point_cloud, const sensor_msgs::PointCloud2 &traced_point_cloud, double match_confidence, double yaw, TransformNeeded &transform_needed, double &match_score, std::string tracer_robot = "", std::string traced_robot = "", std::string current_time = "");

        /*
        Return whether traced image matches tracer image, depending on the confidence.

        If match, output to transform_needed.

        frame_id: "tb3_1/camera_rgb_optical_frame"
        height: 480
        width: 640
        distortion_model: "plumb_bob"
        D: [0.0, 0.0, 0.0, 0.0, 0.0]
        K: [554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0]
        R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        P: [554.254691191187, 0.0, 320.5, -0.0, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 0.0, 1.0, 0.0]
        */
        bool findFurtherTransformNeeded(const cv::Mat &tracer_robot_image, const cv::Mat &traced_robot_image, FeatureType feature_type,
                                        double confidence, double yaw, TransformNeeded &transform_needed, bool &is_unwanted_translation_angle, std::string tracer_robot = "", std::string traced_robot = "", std::string current_time = "");

    private:
        CameraImageProcessor();

        // In order to synchronize image and point cloud although they don't really do
        std::unordered_map<std::string, sensor_msgs::Image> robots_to_temp_image_;

        std::unordered_map<std::string, sensor_msgs::Image> robots_to_current_image_;
        std::unordered_map<std::string, geometry_msgs::Pose> robots_to_current_pose_;
        std::unordered_map<std::string, std::list<PoseImagePair>> robots_to_all_pose_image_pairs_;
        std::unordered_map<std::string, std::unordered_set<size_t>> robots_to_all_visited_pose_image_pair_indexes_;

        std::unordered_map<std::string, sensor_msgs::PointCloud2> robots_to_current_point_cloud_;

        cv::Vec3d rotationMatrixToEulerAngles(cv::Mat &R);

        /*----------ICP parameters------------*/
        double _leaf_size;                               // leaf size for voxel grid
        double _minX, _maxX, _minY, _maxY, _minZ, _maxZ; // min and max pts for box filter
        int _mean_k;                                     // number of neighbors to analyze for each point for noise removal
        double _std_mul;                                 // standard deviation multiplication threshold for noise removal

        double _dist_threshold; // distance threshold for RASNSAC to consider a point as inlier (for ground removal)
        int _eps_angle;         // allowed difference of angles in degrees for perpendicular plane model

        double _transformation_epsilon;    // minimum transformation difference for termination condition
        int _max_iters;                    // max number of registration iterations
        double _euclidean_fitness_epsilon; // maximum allowed Euclidean error between two consecutive steps in the ICP loop
        double _max_correspondence_distance;
        // correspondences with higher distances will be ignored
        void removeInvalidValues(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_ptr);
        void cropCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr);       // crops cloud using box filter
        void removeNoise(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr);     // removes noise using Statistical outlier removal
        void downsampleCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr); // downsampling the point cloud using Voxelgrid
        void removeGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr ground_plane_ptr);                                                     // ground removal using RANSAC
        void filterCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr); // filtering the point cloud
    };
}
#endif