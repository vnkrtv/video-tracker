#include "multitracker.hpp"

namespace detector {

    MultiTracker::MultiTracker(const double &minTrackingQuality): _minTrackingQuality(minTrackingQuality) {
        _objTrackers = map<int, dlib::correlation_tracker>();
        _objLabels = map<int, string>();
        _currentObjID = 0;
    }

    void MultiTracker::update(const dlib::cv_image<dlib::bgr_pixel> &img, SpeedDetector &speedDetector) {
        vector<int> objIDsToDelete;
        for (auto &[objID, tracker]: _objTrackers) {
            double trackingQuality = tracker.update(img);
            if (trackingQuality < _minTrackingQuality) {
                objIDsToDelete.emplace_back(objID);
            } else {
                auto bbox = getObjectBbox(tracker);
                speedDetector.addObject(objID, bbox);
            }
        }
        for (auto &objID: objIDsToDelete) {
            std::clog << "Remove tracker with ID: " << objID << " from list of getTrackers" << std::endl;
            _objTrackers.erase(objID);
        }
    }

    void MultiTracker::addTrackers(const dlib::cv_image<dlib::bgr_pixel> &img,
                                   const vector<DetectionResult> &detectedObjects) {
        for (auto &obj : detectedObjects) {
            auto bbox = obj.bbox;
            int x = bbox.x;
            int y = bbox.y;
            int width = bbox.width;
            int height = bbox.height;

            int xBar = x + static_cast<int>(0.5 * width);
            int yBar = y + static_cast<int>(0.5 * height);

            int matchObjID = -1;
            for (auto &[objID, tracker]: _objTrackers) {
                auto trackedPosition = tracker.get_position();

                int tx = static_cast<int>(trackedPosition.left());
                int ty = static_cast<int>(trackedPosition.top());
                int tWidth = static_cast<int>(trackedPosition.width());
                int tHeight = static_cast<int>(trackedPosition.height());

                int txBar = tx + static_cast<int>(0.5 * tWidth);
                int tyBar = ty + static_cast<int>(0.5 * tHeight);

                bool pred = (tx <= xBar) && (xBar <= (tx + tWidth)) &&
                            (ty <= yBar) && (yBar <= (ty + tHeight)) &&
                            (x <= txBar) && (txBar <= (x + width)) &&
                            (y <= tyBar) && (tyBar <= (y + height));
                if (pred) {
                    matchObjID = objID;
                }
            }
            if (matchObjID == -1) {
                std::clog << "Create new tracker: ID(" << _currentObjID << ")" << std::endl;
                dlib::correlation_tracker tracker;
                tracker.start_track(img, dlib::rectangle(x, y, x + width, y + height));
                _objTrackers[_currentObjID] = tracker;
                _objLabels[_currentObjID] = obj.getLabel();
                matchObjID = _currentObjID;
                _currentObjID++;
            }
        }
    }

    [[nodiscard]] cv::Rect2i MultiTracker::getObjectBbox(const dlib::correlation_tracker &tracker) const {
        auto trackedPosition = tracker.get_position();

        int tx = static_cast<int>(trackedPosition.left());
        int ty = static_cast<int>(trackedPosition.top());
        int tWidth = static_cast<int>(trackedPosition.width());
        int tHeight = static_cast<int>(trackedPosition.height());

        return cv::Rect2i(tx, ty, tWidth, tHeight);
    }

    [[nodiscard]] map<int, dlib::correlation_tracker> MultiTracker::getTrackers() const {
        return _objTrackers;
    }

    [[nodiscard]] string MultiTracker::getLabel(const int &objID) {
        return _objLabels[objID];
    }

} // namespace detector