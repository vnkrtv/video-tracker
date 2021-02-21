#include <map>
#include <unordered_map>
#include <utility>

#include "model.hpp"

//#define USE_TAXICAB_SQRT

namespace detector {

    using std::map;

    struct DetectedObject {

        cv::Point2i centroid;
        cv::Rect2i bbox;

        explicit DetectedObject(cv::Rect2i bbox);

    };

    class SpeedDetector {
    private:

        unordered_map<int, vector<DetectedObject>> _detectedObjects;

        static double getDist(const int &x1, const int &x2, const int &y1, const int &y2);

        static double estimateSpeed(const cv::Point2i &prevLoc, const cv::Point2i &curLoc, const cv::Rect2i &bbox, const double &fps);

    public:

        explicit SpeedDetector();

        void addObject(const int &objID, const cv::Rect2i &objBbox);

        map<int, double> getObjectsSpeed(const double &fps);

    };

} // namespace detector
