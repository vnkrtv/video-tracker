#include "model.hpp"

namespace detector {

    int objectCounter = 0;

    enum TrackerType {
        MIL = 0,
        GOTURN,
    };

    class UnknownTrackerType : public std::exception {
    public:
        virtual const char *what() noexcept {
            return "unknown tracker type. Available types: MIL, GOTURN";
        }
    };

    string fps2str(const double &);

    class ObjectTracker {
    private:

        cv::Ptr<cv::Tracker> _tracker;

        double _timer;
        double _fps;

        int _objectID;

    public:

        explicit ObjectTracker(TrackerType);

        void init(cv::Mat &, cv::Rect2i &);

        bool update(cv::Mat &, cv::Rect2i &);

        bool track(cv::Mat &, cv::Rect2i &);

        void displayFPS(cv::Mat &) const;

    };

} // namespace detector