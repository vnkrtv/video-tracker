#include "centroid_tracker.hpp"

namespace detector {

    double getDist(int x1, int x2, int y1, int y2) {
#ifdef USE_TAXICAB_SQRT
        return double(abs(x2 - x1) + abs(y2 - y1));
#else
        return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
    }
#endif

    vector<vector<double>> getDistMatrix(vector<cv::Point2i> & objectCentroids, vector<cv::Point2i> & inputCentroids) {
        vector<vector<double>> distVec(objectCentroids.size());
        for (int i = 0; i < objectCentroids.size(); i++) {
            distVec[i] = vector<double>(inputCentroids.size());
            for(int t = 0; t < inputCentroids.size(); t++) {
                distVec[i][t] = getDist(objectCentroids[i].x, inputCentroids[t].x,
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

    vector<int> sortIndexesByElements(const vector<double> &v) {
        vector<int> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
            return v[i1] < v[i2];
        });

        return idx;
    }

    string fps2str(const double & fps) {
        return dynamic_cast< std::ostringstream & >(
                (std::ostringstream() << std::dec << static_cast<int>(fps))
        ).str();
    }

    CentroidTracker::CentroidTracker(int maxDisappeared): _maxDisappeared(maxDisappeared) {
        _nextObjectId = 0;
        _objects = map<int, cv::Point2i>();
        _disappeared = map<int, int>();
    }

    void CentroidTracker::registerObject(cv::Point2i & centroid) {
        _objects[_nextObjectId] = centroid;
        _disappeared[_nextObjectId] = 0;
        _nextObjectId++;
    }

    void CentroidTracker::deregisterObject(int objectId) {
        _objects.erase(objectId);
        _disappeared.erase(objectId);
    }

    map<int, cv::Point2i> CentroidTracker::update(vector<cv::Rect2i> & rectsVec) {
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
            auto rowsVec = sortIndexesByElements(minVec);
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


} // namespace detector