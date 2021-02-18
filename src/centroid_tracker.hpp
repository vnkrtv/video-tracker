#include "object_tracker.hpp"

//#define USE_TAXICAB_SQRT

namespace detector {

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

} // namespace detector