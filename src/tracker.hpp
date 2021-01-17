#include <map>
#include <numeric>
#include <algorithm>

#include "model.hpp"


#define USE_TAXICAB_SQRT

namespace detector {

    using std::map;

//    int objectCounter = 0;

    enum TrackerType {
        TR_MIL = 0,
        TR_GOTURN,
    };

    double getEuclideanDist(int , int , int , int);

    class CentroidTracker {
    private:

        int _nextObjectId;
        map<int, cv::Point2i> _objects;
        map<int, int> _disappeared;
        int _maxDisappeared;

        vector<vector<double>> getDistMatrix(vector<cv::Point2i> & objectCentroids, vector<cv::Point2i> & inputCentroids) {
            vector<vector<double>> distVec(objectCentroids.size());
            for (int i = 0; i < objectCentroids.size(); i++) {
                distVec[i] = vector<double>(inputCentroids.size());
                for(int t = 0; t < inputCentroids.size(); t++) {
                    distVec[i][t] = getEuclideanDist(objectCentroids[i].x, inputCentroids[t].x,
                                                     objectCentroids[i].y, inputCentroids[t].y);
                }
            }
            return distVec;
        }

        vector<double> getSortedMinRowElementsVec(vector<vector<double>> & distMatrix, vector<int> & minElementsIdxVec) {
            vector<double> minVec(distMatrix.size());
            for (auto & row: distMatrix) {
                auto minElementIt = std::min_element(row.begin(), row.end());
                minVec.emplace_back(*minElementIt);
                minElementsIdxVec.emplace_back(std::distance(row.begin(), minElementIt));
            }
            std::sort(minVec.begin(), minVec.end());
            return minVec;
        }

        vector<int> sortIndexes(const vector<double> &v) {

            // initialize original index locations
            vector<int> idx(v.size());
            std::iota(idx.begin(), idx.end(), 0);

            // sort indexes based on comparing values in v
            // using std::stable_sort instead of std::sort
            // to avoid unnecessary index re-orderings
            // when v contains elements of equal values
            stable_sort(idx.begin(), idx.end(),
                        [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

            return idx;
        }

    public:

        CentroidTracker(int maxDisappeared): _maxDisappeared(maxDisappeared) {
            _nextObjectId = 0;
            _objects = map<int, cv::Point2i>();
            _disappeared = map<int, int>();
        }

        void registerObject(cv::Point2i & centroid) {
            _objects[_nextObjectId] = centroid;
            _disappeared[_nextObjectId] = 0;
            _nextObjectId++;
        }

        void deregisterObject(int objectId) {
            _objects.erase(_nextObjectId);
            _disappeared.erase(_nextObjectId);
        }

        map<int, cv::Point2i> update(vector<cv::Rect2i> & rectsVec) {
            if (rectsVec.empty()) {
                for (auto & [objectID, framesCount]: _disappeared) {
                    _disappeared[objectID]++;
                    if (_disappeared[objectID] > _maxDisappeared) {
                        deregisterObject(objectID);
                    }
                }
                return _objects;
            }

            vector<cv::Point2i> inputCentroids(rectsVec.size());
            for (int i = 0; i < rectsVec.size(); i++) {
                auto bbox = rectsVec[i];
                int cX = int((2 * bbox.x + bbox.width) / 2);
                int cY = int((2 * bbox.y + bbox.height) / 2);
                inputCentroids[i] = cv::Point2i(cX, cY);
            }

            if (_objects.empty()) {
                for (auto & centroid : inputCentroids) {
                    registerObject(centroid);
                }
            } else {
                vector<int> objectIds;
                vector<cv::Point2i> objectCentroids;
                for (auto & [objectId, centroid]: _objects) {
                    objectIds.emplace_back(objectId);
                    objectCentroids.emplace_back(centroid);
                }

                auto distMatrix = getDistMatrix(objectCentroids, inputCentroids);
                vector<int> minElementsIdxVec;
                auto minVec = getSortedMinRowElementsVec(distMatrix, minElementsIdxVec);
                auto rowsVec = sortIndexes(minVec);
                auto colsVec = vector<int>(minElementsIdxVec.size());
                for (int i = 0; i < minElementsIdxVec.size(); i++) {
                    colsVec[i] = colsVec[minElementsIdxVec[i]];
                }

                set<int> usedRows;
                set<int> usedCols;
                set<int> allRows;
                set<int> allCols;

                for (int i = 0; i < objectCentroids.size(); i++) {
                    allRows.insert(i);
                }
                for (int i = 0; i < inputCentroids.size(); i++) {
                    allCols.insert(i);
                }

                for (int i = 0; i < colsVec.size(); i++) {
                    int row = rowsVec[i];
                    int col = colsVec[i];

                    if (usedRows.find(row) != usedRows.end() || usedCols.find(col) != usedCols.end()) {
                        continue;
                    }

                    int objectId = objectIds[row];
                    _objects[objectId] = inputCentroids[col];
                    _disappeared[objectId] = 0;

                    usedRows.insert(row);
                    usedCols.insert(col);

                    std::set<int> unusedRows;
                    std::set_difference(allRows.begin(), allRows.end(), usedRows.begin(), usedRows.end(), std::inserter(unusedRows, unusedRows.end()));

                    std::set<int> unusedCols;
                    std::set_difference(allCols.begin(), allCols.end(), usedCols.begin(), usedCols.end(), std::inserter(unusedCols, unusedCols.end()));

                    if (objectCentroids.size() > inputCentroids.size()) {
                        for (auto & unusedRow : unusedRows) {
                            objectId = objectIds[unusedRow];
                            _disappeared[objectId]++;

                            if (_disappeared[objectId] > _maxDisappeared) {
                                deregisterObject(objectId);
                            }
                        }
                    } else {
                        for (auto & unusedCol : unusedCols) {
                            registerObject(inputCentroids[unusedCol]);
                        }
                    }
                }
            }
            return _objects;
        }

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