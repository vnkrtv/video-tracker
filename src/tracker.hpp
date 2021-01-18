#include <map>
#include <numeric>
#include <algorithm>

#include "model.hpp"


//#define USE_TAXICAB_SQRT

namespace detector {

    using std::map;

    enum TrackerType {
        TR_MIL = 0,
        TR_GOTURN,
    };

    double getDist(int, int, int, int);

    vector<vector<double>> getDistMatrix(vector<cv::Point2i> &, vector<cv::Point2i> &);

    vector<double> getSortedMinRowElementsVec(vector<vector<double>> &, vector<int> &);

    vector<int> sortIndexesByElements(const vector<double> &);

    class CentroidTracker {
    private:

        int _nextObjectId;
        map<int, cv::Point2i> _objects;
        map<int, int> _disappeared;
        int _maxDisappeared;

    public:

        explicit CentroidTracker(int);

        void registerObject(cv::Point2i &);

        void deregisterObject(int);

        map<int, cv::Point2i> update(vector<cv::Rect2i> &);

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