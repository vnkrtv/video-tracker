#include <map>
#include <stack>
#include <unordered_map>
#include <utility>

#include "model.hpp"

//#define USE_TAXICAB_SQRT

namespace detector {

    using std::map;
    using std::queue;

    struct DetectedObject {

        cv::Point2i centroid;
        cv::Rect2i bbox;
        float meanWidth;

        explicit DetectedObject(cv::Rect2i bbox, const int &objClass);

    };

    class SpeedDetector {
    private:

        unordered_map<int, queue<DetectedObject>> _detectedObjects;

        static double getDist(const int &x1, const int &x2, const int &y1, const int &y2);

        static double estimateSpeed(const cv::Point2i &prevLoc,
                                    const cv::Point2i &curLoc,
                                    const cv::Rect2i &bbox,
                                    const float &meanObjWidth,
                                    const double &fps);

    public:

        explicit SpeedDetector();

        void addObject(const int &objID, const cv::Rect2i &objBbox, const int &objClass);

        map<int, double> getObjectsSpeed(const double &fps);

    };

} // namespace detector
