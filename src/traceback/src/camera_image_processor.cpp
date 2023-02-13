#include <traceback/camera_image_processor.h>

#include <ros/assert.h>
#include <ros/console.h>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include <fstream>

namespace traceback
{
    CameraImageProcessor::CameraImageProcessor()
    {
        _leaf_size = 0.025;
        // _dist_threshold = 0.1;
        // _eps_angle = 15;
        // _minX = 0.0;
        // _minY = -25.0;
        // _minZ = -3.0;
        // _maxX = 50.0; // default 40.0
        // _maxY = 25.0;
        // _maxZ = 3.0;
        _mean_k = 50;
        _std_mul = 1.5;                       // default 1.0
        _transformation_epsilon = 0.00000001; // default 0.01
        _max_iters = 75;
        _euclidean_fitness_epsilon = 1.0;
        _max_correspondence_distance = 3.0; // default 1.0
    }

    bool CameraImageProcessor::pointCloudMatching(const sensor_msgs::PointCloud2 &tracer_point_cloud, const sensor_msgs::PointCloud2 &traced_point_cloud, double match_confidence, double yaw, TransformNeeded &transform_needed, double &match_score, std::string tracer_robot, std::string traced_robot, std::string current_time)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tracer_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr traced_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::fromROSMsg(tracer_point_cloud, *tracer_cloud_ptr);
        filterCloud(tracer_cloud_ptr, tracer_cloud_ptr);
        pcl::fromROSMsg(traced_point_cloud, *traced_cloud_ptr);
        filterCloud(traced_cloud_ptr, traced_cloud_ptr);

        if (tracer_cloud_ptr->size() == 0 || traced_cloud_ptr->size() == 0)
        {
            return false;
        }

        // cout<<"Filtered cloud has "<<filtered_cloud_ptr->size()<<"points"<<endl;
        // cout<<"Current cloud has "<<traced_cloud_ptr->size()<<"points"<<endl;

        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setTransformationEpsilon(_transformation_epsilon);
        icp.setMaximumIterations(_max_iters);
        icp.setMaxCorrespondenceDistance(_max_correspondence_distance);
        icp.setEuclideanFitnessEpsilon(_euclidean_fitness_epsilon);
        // icp.setInputSource(_prev_cloud_ptr);
        // icp.setInputTarget(_downsampled_cloud_ptr);
        icp.setInputSource(traced_cloud_ptr);
        icp.setInputTarget(tracer_cloud_ptr);
        icp.setRANSACOutlierRejectionThreshold(0.025);

        Eigen::AngleAxisf init_rotation(0.0, Eigen::Vector3f::UnitZ());
        Eigen::Translation3f init_translation(0.0, 0.0, 0.0);
        Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();

        // cout<<"-------Matching clouds---------"<<endl;
        icp.align(*transformed_cloud_ptr, init_guess);

        // cout<<"-------Matching done---------"<<endl;

        if (!icp.hasConverged())
        {
            // {
            //     std::ofstream fw(current_time + "_ICP_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::app);
            //     if (fw.is_open())
            //     {
            //         fw << "ICP does not converge." << std::endl;
            //         fw.close();
            //     }
            // }
            return false;
        }

        Eigen::Matrix4f t = icp.getFinalTransformation();
        Eigen::Matrix4f prev_transformation = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f curr_transformation = prev_transformation * t; // final transformation matrix

        Eigen::Matrix3f mat;   // rotation matrix
        Eigen::Vector3f trans; // translation vector

        trans << curr_transformation(0, 3), curr_transformation(1, 3), curr_transformation(2, 3);
        mat << curr_transformation(0, 0), curr_transformation(0, 1), curr_transformation(0, 2),
            curr_transformation(1, 0), curr_transformation(1, 1), curr_transformation(1, 2),
            curr_transformation(2, 0), curr_transformation(2, 1), curr_transformation(2, 2);

        Eigen::Quaternionf quat(mat); // rotation matrix stored as a quaternion

        double score = icp.getFitnessScore();
        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + "_ICP_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot.txt", std::ofstream::out);
            if (fw.is_open())
            {
                fw << "ICP has converged." << std::endl;
                fw << "Score: " << score << std::endl;
                fw << "Transformation:" << std::endl;
                fw << "position(x, y, z) = (" << trans[0] << ", " << trans[1] << ", " << trans[2] << ")" << std::endl;
                fw << "rotation(x, y, z, w) = (" << quat.x() << ", " << quat.y() << ", " << quat.z() << ", " << quat.w() << ")" << std::endl;
                fw.close();
            }
        }

        /*
        Bear in mind that tracer is the src image and traced is the dst image.

        transform_t is of the form (x, y, z)
        t.y should be considered 0.
        If yaw=0, the more positive the t.z is, the more the tracer is behind from the goal (traced), the more positive x translation is needed
        to translate from tracer to traced. This is different from using image matching.
        If yaw=0, the more positive the t.x is, the more the tracer is to the left of the goal (traced), the more negative y translation is needed
        to translate from tracer to traced.

        For transform_R, only consider the y-axis rotation since this y-axis rotation is the z-axis rotation in the robot world.
        The more positive this rotation, the more negative z-axis rotation is needed in the robot world.

        transform_R can be directly used.
        */

        // Read the above comment to understand these calculations.
        // Note the sign of the effect of trans[0] and trans[2].
        // It's quite complicated to figure it out.
        transform_needed.tx = trans[2] * cos(yaw) + trans[0] * sin(yaw);
        transform_needed.ty = (-1 * trans[0] * cos(yaw)) + trans[2] * sin(yaw);
        transform_needed.r = quat.y() * -1.0;

        match_score = score;

        if (score < match_confidence)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void CameraImageProcessor::removeInvalidValues(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_ptr)
    {
        pcl::PointCloud<pcl::PointXYZ>::iterator it = cloud_ptr->points.begin();
        while (it != cloud_ptr->points.end())
        {
            double x = it->x;
            double y = it->y;
            double z = it->z;
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            {
                it = cloud_ptr->points.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    /* @brief Cropping the cloud using Box filter */
    void CameraImageProcessor::cropCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr)
    {
        pcl::CropBox<pcl::PointXYZ> boxFilter;
        boxFilter.setMin(Eigen::Vector4f(_minX, _minY, _minZ, 1.0));
        boxFilter.setMax(Eigen::Vector4f(_maxX, _maxY, _maxZ, 1.0));
        boxFilter.setInputCloud(in_cloud_ptr);
        boxFilter.filter(*out_cloud_ptr);

        // cout<<"Crop Input: "<<in_cloud_ptr->size()<<" pts, Crop output: "<<out_cloud_ptr->size()<<" pts"<<endl;

        return;
    }

    /* @brief Removing Noise using Statistical outlier method */
    void CameraImageProcessor::removeNoise(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr)
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(in_cloud_ptr);
        sor.setMeanK(_mean_k);
        sor.setStddevMulThresh(_std_mul);
        sor.filter(*out_cloud_ptr);

        // cout<<"Noise Input: "<<in_cloud_ptr->size()<<" pts, Noise output: "<<out_cloud_ptr->size()<<" pts"<<endl;

        return;
    }

    /* @brief Downsampling using Aprroximate Voxel grid filter */
    void CameraImageProcessor::downsampleCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr)
    {
        // cout<<"-------Downsampling cloud---------"<<endl;

        pcl::ApproximateVoxelGrid<pcl::PointXYZ> approx_vg;
        approx_vg.setLeafSize(_leaf_size, _leaf_size, _leaf_size);
        approx_vg.setInputCloud(in_cloud_ptr);
        approx_vg.filter(*out_cloud_ptr);

        // cout<<"DS Input: "<<in_cloud_ptr->size()<<" pts, DS output: "<<out_cloud_ptr->size()<<" pts"<<endl;

        return;
    }

    /* @brief Removes ground plane using perpendicular plane model with RANSAC */
    void CameraImageProcessor::removeGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr,
                                            pcl::PointCloud<pcl::PointXYZ>::Ptr ground_plane_ptr)
    {
        // cout<<"-------Removing ground---------"<<endl;

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        // Creating the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true); // optional
        seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(_dist_threshold);
        seg.setAxis(Eigen::Vector3f(0, 0, 1)); // z-axis
        seg.setEpsAngle(_eps_angle);
        seg.setInputCloud(in_cloud_ptr);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0)
        {
            std::cout << "Could not estimate the plane" << std::endl;
        }

        // Remove ground from the cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(in_cloud_ptr);
        extract.setIndices(inliers);
        extract.setNegative(true); // true removes the indices
        extract.filter(*out_cloud_ptr);

        // Extract ground from the cloud
        extract.setNegative(false); // false leaves only the indices
        extract.filter(*ground_plane_ptr);

        // cout<<"GR Input: "<<in_cloud_ptr->size()<<" pts, GR output: "<<out_cloud_ptr->size()<<" pts"<<endl;

        return;
    }

    /* @brief Filters the point cloud using cropping, ground and noise removal filters and then downsamples */
    void CameraImageProcessor::filterCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr no_ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr only_ground_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr no_noise_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        // pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

        // ROS_DEBUG("debug");
        // cropCloud(in_cloud_ptr, cropped_cloud_ptr);
        // ROS_DEBUG("debug");
        // removeGround(cropped_cloud_ptr, no_ground_cloud_ptr, only_ground_cloud_ptr);
        ROS_DEBUG("Before removeNoise, size: %zu", in_cloud_ptr->points.size());
        removeNoise(in_cloud_ptr, no_noise_cloud_ptr);
        // removeNoise(no_ground_cloud_ptr, out_cloud_ptr);
        ROS_DEBUG("Before downsampleCloud, size: %zu", no_noise_cloud_ptr->points.size());
        downsampleCloud(no_noise_cloud_ptr, out_cloud_ptr);
        ROS_DEBUG("Before removeInvalidValues, size: %zu", out_cloud_ptr->points.size());
        removeInvalidValues(out_cloud_ptr);
        ROS_DEBUG("After removeInvalidValues, size: %zu", out_cloud_ptr->points.size());

        return;
    }

    bool CameraImageProcessor::matchImage(const cv::Mat &tracer_robot_image, const cv::Mat &traced_robot_image, FeatureType feature_type,
                                          double confidence, std::string tracer_robot, std::string traced_robot, std::string current_time)
    {
        const std::vector<cv::Mat> &images = {tracer_robot_image, traced_robot_image};
        std::vector<cv::detail::ImageFeatures> image_features;
        std::vector<cv::detail::MatchesInfo> pairwise_matches;
        std::vector<cv::detail::CameraParams> transforms;
        std::vector<int> good_indices;
        // TODO investigate value translation effect on features
        auto finder = internal::chooseFeatureFinder(feature_type);
        cv::Ptr<cv::detail::FeaturesMatcher> matcher =
            cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>();

        if (tracer_robot_image.empty() || traced_robot_image.empty())
        {
            ROS_ERROR("Either traced robot image or tracer robot image is empty, which should not be the case!");
            return false;
        }

        image_features.reserve(images.size());
        for (const cv::Mat &image : images)
        {
            image_features.emplace_back();
            if (!image.empty())
            {
#if CV_VERSION_MAJOR >= 4
                cv::detail::computeImageFeatures(finder, image, image_features.back());
#else
                (*finder)(image, image_features.back());
#endif
            }
        }
        finder = {};

        /* find corespondent features */
        try
        {
            ROS_DEBUG("pairwise matching features");
            (*matcher)(image_features, pairwise_matches);
            matcher = {};
        }
        catch (std::exception e)
        {
            ROS_INFO("Not enough features, catched!");
            return false;
        }

        // #ifndef NDEBUG
        internal::writeDebugMatchingInfo(images, image_features, pairwise_matches, traced_robot, tracer_robot, current_time);
        // #endif

        double match_score = -1.0; // does not even have match
        for (auto &match_info : pairwise_matches)
        {
            if (match_info.H.empty() ||
                match_info.src_img_idx >= match_info.dst_img_idx)
            {
                continue;
            }
            match_score = match_info.confidence;
        }

        good_indices = cv::detail::leaveBiggestComponent(
            image_features, pairwise_matches, static_cast<float>(confidence));

        // Write match score to file
        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + "Match_score_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot" + "_match_score.txt", std::ofstream::app);
            if (fw.is_open())
            {
                if (good_indices.size() == 1)
                {
                    fw << "Does not match at time " << current_time << " with confidence " << match_score << std::endl;
                }
                else
                {
                    fw << "Match at time " << current_time << " with confidence " << match_score << std::endl;
                }
                fw.close();
            }
        }
        // END

        // no match found
        if (good_indices.size() == 1)
        {
            return false;
        }

        return true;
    }

    bool CameraImageProcessor::findFurtherTransformNeeded(const cv::Mat &tracer_robot_image, const cv::Mat &traced_robot_image, FeatureType feature_type,
                                                          double confidence, double yaw, TransformNeeded &transform_needed, bool &is_unwanted_translation_angle, std::string tracer_robot, std::string traced_robot, std::string current_time)
    {
        const std::vector<cv::Mat> &images = {tracer_robot_image, traced_robot_image};
        std::vector<cv::detail::ImageFeatures> image_features;
        std::vector<cv::detail::MatchesInfo> pairwise_matches;
        std::vector<cv::detail::CameraParams> transforms;
        std::vector<int> good_indices;
        // TODO investigate value translation effect on features
        auto finder = internal::chooseFeatureFinder(feature_type);
        cv::Ptr<cv::detail::FeaturesMatcher> matcher =
            cv::makePtr<cv::detail::AffineBestOf2NearestMatcher>();

        if (tracer_robot_image.empty() || traced_robot_image.empty())
        {
            ROS_ERROR("Either traced robot image or tracer robot image is empty, which should not be the case!");
            return false;
        }

        ROS_DEBUG("findFurtherTransformNeeded computing features");
        image_features.reserve(images.size());
        for (const cv::Mat &image : images)
        {
            image_features.emplace_back();
            if (!image.empty())
            {
#if CV_VERSION_MAJOR >= 4
                cv::detail::computeImageFeatures(finder, image, image_features.back());
#else
                (*finder)(image, image_features.back());
#endif
            }
        }
        finder = {};

        /* find corespondent features */
        try
        {
            ROS_DEBUG("pairwise matching features");
            (*matcher)(image_features, pairwise_matches);
            matcher = {};
        }
        catch (std::exception e)
        {
            ROS_INFO("Not enough features, catched!");
            return false;
        }

        // #ifndef NDEBUG
        internal::writeDebugMatchingInfo(images, image_features, pairwise_matches, traced_robot, tracer_robot, current_time);
        // #endif

        double match_score = -1.0; // suppose to be changed
        for (auto &match_info : pairwise_matches)
        {
            if (match_info.H.empty() ||
                match_info.src_img_idx >= match_info.dst_img_idx)
            {
                continue;
            }
            match_score = match_info.confidence;
        }

        /* use only matches that has enough confidence. leave out matches that are not
         * connected (small components) */
        /* e.g. pairwise_matches becomes [(0, 0), (0, 1), (1, 0), (1, 1)]
        good_indices becomes [1, 2]
        Therefore, the 0 and 1 in pairwise_matches actually correspond to images 1 and 2 rather than images 0 and 1,
        so do the transforms 0 and 1 later, which actually correspond to images 1 and 2 too. */
        good_indices = cv::detail::leaveBiggestComponent(
            image_features, pairwise_matches, static_cast<float>(confidence));

        // Write match score to file
        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + "Match_score_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot" + "_match_score.txt", std::ofstream::app);
            if (fw.is_open())
            {
                if (good_indices.size() == 1)
                {
                    fw << "Does not match at time " << current_time << " with confidence " << match_score << std::endl;
                }
                else
                {
                    fw << "Match at time " << current_time << " with confidence " << match_score << std::endl;
                }
                fw.close();
            }
        }
        // END

        // no match found
        if (good_indices.size() == 1)
        {
            return false;
        }

        for (auto &match_info : pairwise_matches)
        {
            ROS_INFO("match_info %d, %d", match_info.src_img_idx, match_info.dst_img_idx);
        }
        for (auto &indice : good_indices)
        {
            ROS_INFO("indice %d", indice);
        }
        // ROS_INFO("transforms size %zu", transforms.size());
        ROS_INFO("Debug");

        std::vector<cv::Point2f> points1, points2;

        if (image_features.size() == 2)
        {
            // std::vector<cv::KeyPoint> keypoints1 = image_features[0].keypoints;
            // cv::KeyPoint::convert(keypoints1, points1);
            // std::vector<cv::KeyPoint> keypoints2 = image_features[1].keypoints;
            // cv::KeyPoint::convert(keypoints2, points2);
            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            for (auto &match_info : pairwise_matches)
            {
                if (match_info.H.empty() ||
                    match_info.src_img_idx >= match_info.dst_img_idx)
                {
                    continue;
                }
                for (auto &match : match_info.matches)
                {
                    // match.imgIdx is always 0
                    keypoints1.emplace_back(image_features[0].keypoints[match.queryIdx]);
                    keypoints2.emplace_back(image_features[1].keypoints[match.trainIdx]);
                }
            }
            cv::KeyPoint::convert(keypoints1, points1);
            cv::KeyPoint::convert(keypoints2, points2);
        }
        else
        {
            ROS_ERROR("image_features must be of size 2, matching exactly 2 images!");
        }

        double k[9] = {554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0};
        cv::Mat camera_K = cv::Mat(3, 3, CV_64F, k);

        cv::Mat essential_mat = cv::findEssentialMat(points1, points2, camera_K, cv::RANSAC);

        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot" + "_essential_mat.txt", std::ofstream::out);
            if (fw.is_open())
            {
                fw << "Camera calibration matrix K:" << std::endl;
                fw << camera_K.at<double>(0, 0) << "\t" << camera_K.at<double>(0, 1) << "\t" << camera_K.at<double>(0, 2) << std::endl;
                fw << camera_K.at<double>(1, 0) << "\t" << camera_K.at<double>(1, 1) << "\t" << camera_K.at<double>(1, 2) << std::endl;
                fw << camera_K.at<double>(2, 0) << "\t" << camera_K.at<double>(2, 1) << "\t" << camera_K.at<double>(2, 2) << std::endl;
                fw << "Essential matrix E:" << std::endl;
                fw << essential_mat.at<double>(0, 0) << "\t" << essential_mat.at<double>(0, 1) << "\t" << essential_mat.at<double>(0, 2) << std::endl;
                fw << essential_mat.at<double>(1, 0) << "\t" << essential_mat.at<double>(1, 1) << "\t" << essential_mat.at<double>(1, 2) << std::endl;
                fw << essential_mat.at<double>(2, 0) << "\t" << essential_mat.at<double>(2, 1) << "\t" << essential_mat.at<double>(2, 2) << std::endl;
                fw.close();
            }
        }

        /*
        Bear in mind that tracer is the src image and traced is the dst image.

        transform_t is of the form (x, y, z)
        t.y should be considered 0.
        If yaw=0, the more positive the t.z is, the more the goal (traced) is behind from the tracer, the more negative x translation is needed
        to translate from tracer to traced.
        If yaw=0, the more positive the t.x is, the more the goal (traced) is to the left of the tracer, the more positive y translation is needed
        to translate from tracer to traced.

        When computing transform_t, must consider the current orientation.
        The goal is already in the tracer's coordinate frame, and the essential matrix
        is actually to rotate, then translate from tracer to traced, in the tracer's coordinate frame.

        transform_t probably needs to be scaled (element-wise multiplication) before it can be used.

        For transform_R, only consider the y-axis rotation since this y-axis rotation is the z-axis rotation in the robot world.
        The more positive this rotation, the more positive z-axis rotation is needed in the robot world.

        transform_R can be directly used.
        */
        cv::Mat transform_R, transform_t;
        int number_of_inliers = cv::recoverPose(essential_mat, points1, points2, camera_K, transform_R, transform_t);

        cv::Vec3d rot = rotationMatrixToEulerAngles(transform_R);

        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot" + "_transform_R.txt", std::ofstream::out);
            if (fw.is_open())
            {
                fw << "Number of inliers: " << number_of_inliers << std::endl;
                fw << "XYZ rotation is: " << rot << std::endl;
                fw << "Rotation matrix R:" << std::endl;
                fw << transform_R.at<double>(0, 0) << "\t" << transform_R.at<double>(0, 1) << "\t" << transform_R.at<double>(0, 2) << std::endl;
                fw << transform_R.at<double>(1, 0) << "\t" << transform_R.at<double>(1, 1) << "\t" << transform_R.at<double>(1, 2) << std::endl;
                fw << transform_R.at<double>(2, 0) << "\t" << transform_R.at<double>(2, 1) << "\t" << transform_R.at<double>(2, 2) << std::endl;
                // fw << std::endl;
                fw.close();
            }
        }

        {
            std::ofstream fw(tracer_robot.substr(1) + "_" + traced_robot.substr(1) + "/" + current_time + "_" + tracer_robot.substr(1) + "_tracer_robot_" + traced_robot.substr(1) + "_traced_robot" + "_transform_t.txt", std::ofstream::out);
            if (fw.is_open())
            {
                fw << "Number of inliers: " << number_of_inliers << std::endl;
                fw << "Translation matrix t:" << std::endl;
                fw << transform_t.at<double>(0, 0) << std::endl;
                fw << transform_t.at<double>(1, 0) << std::endl;
                fw << transform_t.at<double>(2, 0) << std::endl;
                fw.close();
            }
        }

        // Read the above comment to understand these calculations.
        // Note the sign of the effect of transform_t.at<double>(0, 0) and transform_t.at<double>(2, 0).
        // It's quite complicated to figure it out.
        transform_needed.tx = (-1 * transform_t.at<double>(2, 0) * cos(yaw)) + (-1 * transform_t.at<double>(0, 0) * sin(yaw));
        transform_needed.ty = transform_t.at<double>(0, 0) * cos(yaw) + (-1 * transform_t.at<double>(2, 0) * sin(yaw));
        transform_needed.r = rot[1];

        // Does not proceed to second traceback for unwanted translation angle to prevent triangulation huge error
        // Unwanted if it is too straight within 0.1 radian difference
        double angle = atan2(transform_t.at<double>(2, 0), transform_t.at<double>(0, 0));
        double PI = 3.1415926;
        if (abs(angle - PI / 2) < 0.1 || abs(angle + PI / 2) < 0.1)
        {
            is_unwanted_translation_angle = true;
        }
        else
        {
            is_unwanted_translation_angle = false;
        }

        ROS_INFO("Debug");

        return true;
    }

    // Calculates rotation matrix to euler angles
    // The result is the same as MATLAB except the order
    // of the euler angles ( x and z are swapped ).
    cv::Vec3d CameraImageProcessor::rotationMatrixToEulerAngles(cv::Mat &R)
    {
        double sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

        bool singular = sy < 1e-6; // If

        double x, y, z;
        if (!singular)
        {
            x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
            y = atan2(-R.at<double>(2, 0), sy);
            z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
        }
        else
        {
            x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
            y = atan2(-R.at<double>(2, 0), sy);
            z = 0;
        }
        return cv::Vec3d(x, y, z);
    }
}
