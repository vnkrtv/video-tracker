#include <map>
#include <unordered_map>
#include <utility>

#include "model.hpp"

//#define USE_TAXICAB_SQRT

namespace detector {

    using std::map;
    using std::unordered_map;

    struct DetectedObject {

        cv::Point2i centroid;
        cv::Rect2i bbox;

        explicit DetectedObject(cv::Rect2i);

    };

    class SpeedDetector {
    private:

        unordered_map<int, vector<DetectedObject>> _detectedObjects;

        static double getDist(const int &, const int &, const int &, const int &);

        static double estimateSpeed(const cv::Point2i &, const cv::Point2i &, const cv::Rect2i &, const double &);

    public:

        explicit SpeedDetector();

        void addObject(const int &, const cv::Rect2i &);

        map<int, double> getObjectsSpeed(const double &);

    };

} // namespace detector
