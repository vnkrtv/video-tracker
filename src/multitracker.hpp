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

        explicit MultiTracker(const double &);

        void update(const dlib::cv_image<dlib::bgr_pixel> &, SpeedDetector &);

        void addTrackers(const dlib::cv_image<dlib::bgr_pixel> &, const vector<DetectionResult> &);

        [[nodiscard]] static cv::Rect2i getObjectBbox(const dlib::correlation_tracker &) ;

        [[nodiscard]] map<int, dlib::correlation_tracker> getTrackers() const;

        [[nodiscard]] string getLabel(const int &);

    };

} // namespace detector