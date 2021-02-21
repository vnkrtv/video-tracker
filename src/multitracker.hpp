#include <numeric>
#include <algorithm>

#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/opencv/cv_image.h>

#include "speed_detector.hpp"

namespace detector {

    using std::map;

    class MultiTracker {
    private:

        map<int, dlib::correlation_tracker> _objTrackers;
        map<int, string> _objLabels;

        double _minTrackingQuality;
        int _currentObjID;

    public:

        explicit MultiTracker(const double &minTrackingQuality);

        void update(const dlib::cv_image<dlib::bgr_pixel> &img, SpeedDetector &speedDetector);

        void addTrackers(const dlib::cv_image<dlib::bgr_pixel> &img, const vector<DetectionResult> &detectedObjects);

        [[nodiscard]] static cv::Rect2i getObjectBbox(const dlib::correlation_tracker &tracker) ;

        [[nodiscard]] map<int, dlib::correlation_tracker> getTrackers() const;

        [[nodiscard]] string getLabel(const int &objID);

    };

} // namespace detector