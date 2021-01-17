#include "tracker.hpp"

namespace detector {

#ifdef USE_TAXICAB_SQRT
    double getEuclideanDist(int x1, int x2, int y1, int y2) {
        return double(abs(x2 - x1) + abs(y2 - y1));
    }
#else
    double getEuclideanDist(int x1, int x2, int y1, int y2) {
        return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
    }
#endif

    string fps2str(const double & fps) {
        return dynamic_cast< std::ostringstream & >(
                (std::ostringstream() << std::dec << static_cast<int>(fps))
        ).str();
    }

    ObjectTracker::ObjectTracker(TrackerType trackerType) {
        if (trackerType == TR_MIL) {
            _tracker = cv::TrackerMIL::create();
        } else if (trackerType == TR_GOTURN) {
            _tracker = cv::TrackerGOTURN::create();
        } else {
            throw UnknownTrackerType();
        }
        _timer = (double) cv::getTickCount();
        _fps = cv::getTickFrequency() / ((double) cv::getTickCount() - _timer);
//        _objectID = objectCounter++;
    }

    void ObjectTracker::init(cv::Mat &frame, cv::Rect2i &bbox) {
        _tracker->init(frame, bbox);
    }

    bool ObjectTracker::update(cv::Mat &frame, cv::Rect2i &bbox) {
        bool ok = _tracker->update(frame, bbox);
        _timer = (double) cv::getTickCount();
        _fps = cv::getTickFrequency() / ((double) cv::getTickCount() - _timer);
        return ok;
    }

    bool ObjectTracker::track(cv::Mat &frame, cv::Rect2i &bbox) {
        if (update(frame, bbox)) {
            cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);
            return true;
        } else {
            cv::putText(frame,
                        "Tracking failure detected",
                        cv::Point(100, 80),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.75,
                        cv::Scalar(0, 0, 255),
                        2);
            return false;
        }
    }

    void ObjectTracker::displayFPS(cv::Mat &frame) const {
        putText(frame,
                "ObjectID: " + std::to_string(_objectID) + " FPS : " + fps2str(_fps),
                cv::Point(100, 50),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                cv::Scalar(50, 170, 50),
                2);
    }

} // namespace detector