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

                fvec(i) = sqrt(pow(transformed_predicted_pose[0] - transformed_actual_pose[0], 2) + pow(transformed_predicted_pose[1] - transformed_actual_pose[1], 2));
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
                // r (angle) estimation
                if (i == 2)
                {
                    xPlus(i) += epsilon / 10;
                }
                else
                {
                    xPlus(i) += epsilon;
                }
                Eigen::VectorXd xMinus(x);
                // r (angle) estimation
                if (i == 2)
                {
                    xMinus(i) -= epsilon / 10;
                }
                else
                {
                    xMinus(i) -= epsilon;
                }

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