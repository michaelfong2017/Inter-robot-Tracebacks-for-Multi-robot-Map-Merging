/*******************************************************************************
 * BSD 3-Clause License
 * 
 * Copyright (c) 2023, Fong Chun Him
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

#ifndef TRANSFORM_ESTIMATOR_H_
#define TRANSFORM_ESTIMATOR_H_

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <boost/thread.hpp>

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
        const float ZERO_ERROR = 0.0001f;

        boost::shared_mutex updates_mutex_;

        cv::detail::ImageFeatures computeFeatures(const cv::Mat &image, FeatureType feature = FeatureType::ORB);
        double matchTwoFeatures(cv::detail::ImageFeatures &features1, cv::detail::ImageFeatures &features2, double confidence = 1.0);
        void setTransformsVectors(std::vector<std::vector<cv::Mat>> transforms_vectors);
        void clearTransformsVectors();
        void setConfidences(std::vector<std::vector<double>> confidences);
        void clearConfidences();

        template <typename InputIt>
        void feed(InputIt grids_begin, InputIt grids_end);
        bool estimateTransforms(FeatureType feature = FeatureType::ORB,
                                double confidence = 1.0);
        std::vector<std::vector<cv::Mat>> getTransformsVectors();
        std::vector<cv::Point2f> getCenters();
        std::vector<std::vector<double>> getConfidences();

        void updateBestTransforms(cv::Mat tracer_to_traced, std::string tracer, std::string traced, std::unordered_map<std::string, std::unordered_map<std::string, cv::Mat>> &best_transforms,
                                  std::unordered_set<std::string> &has_best_transforms);

        void printConfidences(const std::vector<std::vector<double>> confidences);
        void printTransformsVectors(const std::vector<std::vector<cv::Mat>> transforms_vectors);

    private:
        std::vector<nav_msgs::OccupancyGrid::ConstPtr> grids_;
        std::vector<cv::Mat> images_;

        std::vector<std::vector<cv::Mat>> transforms_vectors_;
        std::vector<cv::Point2f> centers_;
        std::vector<std::vector<double>> confidences_;

        cv::Point2f findWeightedCenterOfConvexHulls(cv::Mat image, size_t image_index);
        void findWeightedCenter(std::vector<std::vector<cv::Point>> contoursOrHulls, cv::Point2f &center);

        void findPairwiseConfidences(std::vector<cv::detail::MatchesInfo> pairwise_matches, std::vector<int> good_indices, size_t images_size, std::vector<std::vector<double>> &confidences);
        void toPairwiseTransforms(std::vector<cv::detail::CameraParams> transforms, std::vector<int> good_indices, size_t images_size, std::vector<std::vector<cv::Mat>> &transforms_vectors);
        size_t findIdentityTransform(std::vector<cv::detail::CameraParams> transforms);
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