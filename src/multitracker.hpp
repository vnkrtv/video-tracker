#include <numeric>
#include <algorithm>
#include <thread>

#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/opencv/cv_image.h>

#include "speed_detector.hpp"

namespace detector {

    using std::map;
    using std::thread;

    class MultiTracker {
    private:

        SpeedDetector _speedDetector;

        map<int, dlib::correlation_tracker> _objTrackers;
        map<int, int> _objClasses;
        map<int, string> _objLabels;

        double _minTrackingQuality;
        int _currentObjID;

    public:

        explicit MultiTracker(const double &minTrackingQuality);

        void update(const dlib::cv_image<dlib::bgr_pixel> &img);

        void addTrackers(const dlib::cv_image<dlib::bgr_pixel> &img, const vector<DetectionResult> &detectedObjects);

        [[nodiscard]] static cv::Rect2i getObjectBbox(const dlib::correlation_tracker &tracker);

        [[nodiscard]] map<int, dlib::correlation_tracker> getTrackers() const;

        [[nodiscard]] map<int, double> getObjectsSpeed(const double &fps);

        [[nodiscard]] string getLabel(const int &objID);

    };

//    class ParallelTracker {
//    private:
//
//        unordered_map<int, MultiTracker> _workerTrackers;
//        int _nWorkers;
//        unordered_map<int, int> _workersLoad;
//
//        int getWorker() {
//            int minLoadWorkerID = 0;
//            int minWorkerLoad = 100;
//            for (auto &[workerID, workerLoad]: _workersLoad) {
//                if (workerLoad < minWorkerLoad) {
//                    minLoadWorkerID = workerID;
//                }
//            }
//            return minLoadWorkerID;
//        }
//
//    public:
//
//        explicit ParallelTracker(const int &nWorkers, const double &minTrackingQuality) : _nWorkers(nWorkers) {
//            _workerTrackers = unordered_map<int, MultiTracker>();
//            for (int i = 0; i < nWorkers; i++) {
//                _workerTrackers.insert(std::make_pair(i, MultiTracker(minTrackingQuality)));
//                _workersLoad[i] = 0;
//            }
//        }
//
//        void addTrackers(dlib::cv_image<dlib::bgr_pixel> img, vector<DetectionResult> detectedObjects) {
//
//        }
//
//        void update(dlib::cv_image<dlib::bgr_pixel> img) {
//            auto threads = vector<std::thread>();
//            threads.reserve(_nWorkers);
//            for (int i = 0; i < _nWorkers; i++) {
//                threads.emplace_back(thread(
//                        [](MultiTracker &multiTracker, dlib::cv_image<dlib::bgr_pixel> img) {
//                            multiTracker.update(img);
//                        }, std::ref(_workerTrackers[i]), img)
//                );
//            }
//            for_each(threads.begin(), threads.end(), [&](std::thread &th) {
//                th.join();
//            });
//        }
//
//    };

} // namespace detector