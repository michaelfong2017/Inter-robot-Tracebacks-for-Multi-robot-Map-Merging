#ifndef TRANSFORM_ESTIMATOR_H_
#define TRANSFORM_ESTIMATOR_H_

#include <vector>

#include <geometry_msgs/Transform.h>
#include <nav_msgs/OccupancyGrid.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>

namespace traceback
{
    enum class FeatureType
    {
        AKAZE,
        ORB,
        SURF
    };

    class TransformEstimator
    {
    public:
        template <typename InputIt>
        void feed(InputIt grids_begin, InputIt grids_end);
        bool estimateTransforms(FeatureType feature = FeatureType::ORB,
                                double confidence = 1.0);

    private:
        const float ZERO_ERROR = 0.0001f;

        std::vector<nav_msgs::OccupancyGrid::ConstPtr> grids_;
        std::vector<cv::Mat> images_;
        // std::vector<cv::Mat> transforms_;
        std::vector<std::vector<cv::Mat>> transforms_vectors_;

        void findContours(cv::Mat image, size_t image_index);

        void toPairwiseTransforms(std::vector<cv::detail::CameraParams> transforms, std::vector<int> good_indices, size_t images_size, std::vector<std::vector<cv::Mat>> &transforms_vectors);
        size_t findIdentityTransform(std::vector<cv::detail::CameraParams> transforms);

        void printTransformsVectors(const std::vector<std::vector<cv::Mat>> transforms_vectors);
    };

    template <typename InputIt>
    void TransformEstimator::feed(InputIt grids_begin, InputIt grids_end)
    {
        static_assert(std::is_assignable<nav_msgs::OccupancyGrid::ConstPtr &,
                                         decltype(*grids_begin)>::value,
                      "grids_begin must point to nav_msgs::OccupancyGrid::ConstPtr "
                      "data");

        // we can't reserve anything, because we want to support just InputIt and
        // their guarantee validity for only single-pass algos
        images_.clear();
        grids_.clear();
        for (InputIt it = grids_begin; it != grids_end; ++it)
        {
            if (*it && !(*it)->data.empty())
            {
                grids_.push_back(*it);
                /* convert to opencv images. it creates only a view for opencv and does
                 * not copy or own actual data. */
                images_.emplace_back((*it)->info.height, (*it)->info.width, CV_8UC1,
                                     const_cast<signed char *>((*it)->data.data()));
            }
            else
            {
                grids_.emplace_back();
                images_.emplace_back();
            }
        }
    }
}
#endif