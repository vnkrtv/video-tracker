#include "speed_detector.hpp"

namespace detector {

    DetectedObject::DetectedObject(cv::Rect2i bbox) : bbox(std::move(bbox)) {
        centroid = cv::Point2i(bbox.x + (bbox.width / 2), bbox.y + (bbox.height / 2));
    }

    SpeedDetector::SpeedDetector() {
        _detectedObjects = unordered_map<int, vector<DetectedObject>>();
    }

    void SpeedDetector::addObject(const int &objID, const cv::Rect2i &objBbox) {
        if (_detectedObjects.find(objID) != _detectedObjects.end()) {
            _detectedObjects[objID].emplace_back(DetectedObject(objBbox));
        } else {
            _detectedObjects[objID] = vector<DetectedObject>{DetectedObject(objBbox)};
        }
    }

    double SpeedDetector::getDist(const int &x1, const int &x2, const int &y1, const int &y2) {
#ifdef USE_TAXICAB_SQRT
        return double(abs(x2 - x1) + abs(y2 - y1));
#else
        return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
    }

#endif

    double SpeedDetector::estimateSpeed(const cv::Point2i &prevLoc,
                                        const cv::Point2i &curLoc,
                                        const cv::Rect2i &bbox,
                                        const double &fps) {
        // TODO: nihuy'a ne rabotaet, nado razbirat'sya
        auto dPixels = getDist(prevLoc.x, curLoc.x, prevLoc.y, curLoc.y);
        std::clog << "dPixels: " << dPixels << std::endl;
        auto pixelPerMeter = (bbox.height / bbox.width);
        std::clog << "pixelPerMeter: " << pixelPerMeter << std::endl;
        auto dMeters = dPixels / pixelPerMeter;
        std::clog << "dMeters: " << dMeters << std::endl;
        auto toKmPerHour = 3.6;
        auto speed = dMeters * fps * toKmPerHour;
        return speed;
    }

    map<int, double> SpeedDetector::getObjectsSpeed(const double &fps) {
        map<int, double> objSpeed;
        for (auto&[objID, trackHistory]: _detectedObjects) {
            auto recordsCount = trackHistory.size();
            if (recordsCount >= 2) {
                auto prevLoc = trackHistory[recordsCount - 2].centroid;
                auto curLoc = trackHistory[recordsCount - 1].centroid;
                objSpeed[objID] = estimateSpeed(prevLoc, curLoc, trackHistory[recordsCount - 1].bbox, fps);
            }
        }
        return objSpeed;
    }

} // namespace detector
