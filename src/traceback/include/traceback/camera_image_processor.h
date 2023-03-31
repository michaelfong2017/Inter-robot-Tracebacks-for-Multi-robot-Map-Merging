#ifndef CAMERA_IMAGE_PROCESSOR_H_
#define CAMERA_IMAGE_PROCESSOR_H_

#include <traceback/estimation_internal.h>

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <boost/thread.hpp>

#include <geometry_msgs/Pose.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>

namespace traceback
{
    struct PoseImagePair
    {
        geometry_msgs::Pose pose;
        sensor_msgs::Image image;
        sensor_msgs::Image depth_image;
        int64_t stamp;
        bool operator<(const PoseImagePair &rhs) const
        {
            return stamp < rhs.stamp;
        }
    };

    struct FeaturesDepthsPose
    {
        cv::detail::ImageFeatures features;
        std::vector<double> depths;
        geometry_msgs::Pose pose;
    };

    // In robot world coordinates
    // tx and ty are in meters, r is in radian
    // Any length
    struct TransformNeeded
    {
        double tx;
        double ty;
        double r;
        double arrived_x;
        double arrived_y;
    };

    struct MatchAndSolveResult
    {
        bool match;
        bool solved;
    };

    struct LMFunctor
    {
        // 'm' pairs of (x, f(x))
        Eigen::MatrixXd measuredValues;

        // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
        int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
        {
            // 'x' has dimensions n x 1
            // It contains the current estimates for the parameters.

            // 'fvec' has dimensions m x 1
            // It will contain the error for each data point.

            double aParam = x(0);
            double bParam = x(1);
            double cParam = x(2);

            for (int i = 0; i < values(); i++)
            {
                double xValue = measuredValues(i, 0);
                double yValue = measuredValues(i, 1);
                double txValue = measuredValues(i, 2);
                double tyValue = measuredValues(i, 3);
                double rValue = measuredValues(i, 4);

                Eigen::Vector3d pose(xValue, yValue, 1);

                Eigen::Matrix3d actual_T;
                actual_T << cos(rValue), -sin(rValue), txValue,
                    sin(rValue), cos(rValue), tyValue,
                    0, 0, 1;
                Eigen::Vector3d transformed_actual_pose = actual_T * pose;

                Eigen::Matrix3d predicted_T;
                predicted_T << cos(cParam), -sin(cParam), aParam,
                    sin(cParam), cos(cParam), bParam,
                    0, 0, 1;
                Eigen::Vector3d transformed_predicted_pose = predicted_T * pose;

                double rot_0_0 = cos(rValue) * cos(-1.0 * cParam) - sin(rValue) * sin(-1.0 * cParam);
                double rot_1_0 = sin(rValue) * cos(-1.0 * cParam) + cos(rValue) * sin(-1.0 * cParam);
                double angle_difference = abs(atan2(rot_1_0, rot_0_0));
                double ANGLE_ERROR_MULTIPLIER = 100.0;

                fvec(i) = angle_difference * ANGLE_ERROR_MULTIPLIER + sqrt(pow(transformed_predicted_pose[0] - transformed_actual_pose[0], 2) + pow(transformed_predicted_pose[1] - transformed_actual_pose[1], 2));
                // fvec(i) = sqrt(pow(transformed_predicted_pose[0] - transformed_actual_pose[0], 2) + pow(transformed_predicted_pose[1] - transformed_actual_pose[1], 2));
            }
            return 0;
        }

        // Compute the jacobian of the errors
        int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
        {
            // 'x' has dimensions n x 1
            // It contains the current estimates for the parameters.

            // 'fjac' has dimensions m x n
            // It will contain the jacobian of the errors, calculated numerically in this case.

            double epsilon;
            epsilon = 1e-5f;

            for (int i = 0; i < x.size(); i++)
            {
                Eigen::VectorXd xPlus(x);
                xPlus(i) += epsilon;
                Eigen::VectorXd xMinus(x);
                xMinus(i) -= epsilon;

                Eigen::VectorXd fvecPlus(values());
                operator()(xPlus, fvecPlus);

                Eigen::VectorXd fvecMinus(values());
                operator()(xMinus, fvecMinus);

                Eigen::VectorXd fvecDiff(values());
                fvecDiff = (fvecPlus - fvecMinus) / (2.0f * epsilon);

                // r (angle) estimation
                // if (i == 2)
                // {
                //     fvecDiff *= 15;
                // }
                fjac.block(0, i, values(), 1) = fvecDiff;
            }

            return 0;
        }

        // Number of data points, i.e. values.
        int m;

        // Returns 'm', the number of values.
        int values() const { return m; }

        // The number of parameters, i.e. inputs.
        int n;

        // Returns 'n', the number of inputs.
        int inputs() const { return n; }
    };

    struct LMFunctorGlobal
    {
        // 'm' pairs of (x, f(x))
        Eigen::MatrixXd measuredValues;

        // Compute 'm' errors, one for each data point, for the given parameter values in 'x'
        int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
        {
            // 'x' has dimensions n x 1
            // It contains the current estimates for the parameters.

            // 'fvec' has dimensions m x 1
            // It will contain the error for each data point.

            int num_robot = x.rows() / 3 + 1;

            for (int i = 0; i < values(); i++)
            {
                int fromValue = measuredValues(i, 0);
                int toValue = measuredValues(i, 1);
                double xValue = measuredValues(i, 2);
                double yValue = measuredValues(i, 3);
                double txValue = measuredValues(i, 4);
                double tyValue = measuredValues(i, 5);
                double rValue = measuredValues(i, 6);

                Eigen::Vector3d pose(xValue, yValue, 1);

                Eigen::Matrix3d actual_T;
                actual_T << cos(rValue), -sin(rValue), txValue,
                    sin(rValue), cos(rValue), tyValue,
                    0, 0, 1;
                Eigen::Vector3d transformed_actual_pose = actual_T * pose;

                // if data is 0->1, use 0->1 * 0->0
                // if data is 1->0, use 0->0 * 1->0 (need inv)
                // if data is 0->2, use 0->2 * 0->0
                // if data is 2->0, use 0->0 * 2->0 (need inv)
                // if data is 1->2, use 0->2 * 1->0 (need inv)
                // if data is 2->1, use 0->1 * 2->0 (need inv)
                // therefore,
                // if data is a->b, use 0->b * a->0
                // 0->b is the second transform and a->0 is the first transform
                double first_aParam;
                double first_bParam;
                double first_cParam;
                if (fromValue == 0)
                {
                    first_aParam = 0.0;
                    first_bParam = 0.0;
                    first_cParam = 0.0;
                }
                else
                {
                    first_aParam = x(0 + 3 * (fromValue - 1));
                    first_bParam = x(1 + 3 * (fromValue - 1));
                    first_cParam = x(2 + 3 * (fromValue - 1));
                }
                double second_aParam;
                double second_bParam;
                double second_cParam;
                if (toValue == 0)
                {
                    second_aParam = 0.0;
                    second_bParam = 0.0;
                    second_cParam = 0.0;
                }
                else
                {
                    second_aParam = x(0 + 3 * (toValue - 1));
                    second_bParam = x(1 + 3 * (toValue - 1));
                    second_cParam = x(2 + 3 * (toValue - 1));
                }
                Eigen::Matrix3d inv_first_predicted_T;
                inv_first_predicted_T << cos(first_cParam), -sin(first_cParam), first_aParam,
                    sin(first_cParam), cos(first_cParam), first_bParam,
                    0, 0, 1;
                Eigen::Matrix3d second_predicted_T;
                second_predicted_T << cos(second_cParam), -sin(second_cParam), second_aParam,
                    sin(second_cParam), cos(second_cParam), second_bParam,
                    0, 0, 1;

                Eigen::Matrix3d predicted_T = second_predicted_T * inv_first_predicted_T.inverse();

                Eigen::Vector3d transformed_predicted_pose = predicted_T * pose;

                double cParam = atan2(predicted_T(1, 0), predicted_T(0, 0));
                double rot_0_0 = cos(rValue) * cos(-1.0 * cParam) - sin(rValue) * sin(-1.0 * cParam);
                double rot_1_0 = sin(rValue) * cos(-1.0 * cParam) + cos(rValue) * sin(-1.0 * cParam);
                double angle_difference = abs(atan2(rot_1_0, rot_0_0));
                double ANGLE_ERROR_MULTIPLIER = 100.0;

                fvec(i) = angle_difference * ANGLE_ERROR_MULTIPLIER + sqrt(pow(transformed_predicted_pose[0] - transformed_actual_pose[0], 2) + pow(transformed_predicted_pose[1] - transformed_actual_pose[1], 2));
                // fvec(i) = sqrt(pow(transformed_predicted_pose[0] - transformed_actual_pose[0], 2) + pow(transformed_predicted_pose[1] - transformed_actual_pose[1], 2));
            }
            return 0;
        }

        // Compute the jacobian of the errors
        int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
        {
            // 'x' has dimensions n x 1
            // It contains the current estimates for the parameters.

            // 'fjac' has dimensions m x n
            // It will contain the jacobian of the errors, calculated numerically in this case.

            double epsilon;
            epsilon = 1e-5f;

            for (int i = 0; i < x.size(); i++)
            {
                Eigen::VectorXd xPlus(x);
                xPlus(i) += epsilon;
                Eigen::VectorXd xMinus(x);
                xMinus(i) -= epsilon;

                Eigen::VectorXd fvecPlus(values());
                operator()(xPlus, fvecPlus);

                Eigen::VectorXd fvecMinus(values());
                operator()(xMinus, fvecMinus);

                Eigen::VectorXd fvecDiff(values());
                fvecDiff = (fvecPlus - fvecMinus) / (2.0f * epsilon);

                fjac.block(0, i, values(), 1) = fvecDiff;
            }

            return 0;
        }

        // Number of data points, i.e. values.
        int m;

        // Returns 'm', the number of values.
        int values() const { return m; }

        // The number of parameters, i.e. inputs.
        int n;

        // Returns 'n', the number of inputs.
        int inputs() const { return n; }
    };

    class CameraImageProcessor
    {
    public:
        friend class Traceback;

        bool computeFeaturesAndDepths(cv::detail::ImageFeatures &out_features, std::vector<double> &out_depths, const cv::Mat &image, FeatureType feature_type, const cv::Mat &depth_image);

        MatchAndSolveResult matchAndSolveWithFeaturesAndDepths(cv::detail::ImageFeatures &tracer_robot_features, cv::detail::ImageFeatures &traced_robot_features, std::vector<double> tracer_robot_depths, std::vector<double> traced_robot_depths,
                                                               double confidence, double yaw, TransformNeeded &transform_needed, std::string tracer_robot = "", std::string traced_robot = "", std::string current_time = "");

        MatchAndSolveResult matchAndSolve(const cv::Mat &tracer_robot_color_image, const cv::Mat &traced_robot_color_image, const cv::Mat &tracer_robot_depth_image, const cv::Mat &traced_robot_depth_image, FeatureType feature_type,
                                          double confidence, double yaw, TransformNeeded &transform_needed, std::string tracer_robot = "", std::string traced_robot = "", std::string current_time = "");

        // Return optimized tx, ty and r in order
        std::vector<double> LMOptimize(std::vector<double> x_values,
                                       std::vector<double> y_values,
                                       std::vector<double> tx_values,
                                       std::vector<double> ty_values,
                                       std::vector<double> r_values,
                                       double init_tx,
                                       double init_ty,
                                       double init_r);

        // Return optimized 0->1_tx, 0->1_ty, 0->1_r, 0->2_tx, 0->2_ty, 0->2_r, etc
        std::vector<std::vector<double>> LMOptimizeGlobal(std::vector<int> from_indexes,
                                                          std::vector<int> to_indexes,
                                                          std::vector<double> x_values,
                                                          std::vector<double> y_values,
                                                          std::vector<double> tx_values,
                                                          std::vector<double> ty_values,
                                                          std::vector<double> r_values,
                                                          int num_robot);

    private:
        // In order to synchronize image and depth image although they don't really do
        std::unordered_map<std::string, sensor_msgs::Image> robots_to_temp_image_;

        std::unordered_map<std::string, sensor_msgs::Image> robots_to_current_image_;
        std::unordered_map<std::string, geometry_msgs::Pose> robots_to_current_pose_;
        std::unordered_map<std::string, std::vector<PoseImagePair>> robots_to_all_pose_image_pairs_;
        std::unordered_map<std::string, std::unordered_set<size_t>> robots_to_all_visited_pose_image_pair_indexes_;

        std::unordered_map<std::string, sensor_msgs::Image> robots_to_current_depth_image_;

        cv::Vec3d rotationMatrixToEulerAngles(cv::Mat &R);
    };
}
#endif