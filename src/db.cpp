#include "db.hpp"

namespace detector {

    int callback(void *notUsed, int argc, char **argv, char **azColName) {
        return 0;
    }

    DBException::DBException(string errMessage) : _errMessage(std::move(errMessage)) {}

    const char *DBException::what() noexcept {
        return _errMessage.c_str();
    }

    Storage::Storage(const string &dbFileName) {
        int resCode = sqlite3_open(dbFileName.c_str(), &_db);
        if (resCode) {
            sqlite3_close(_db);
            throw DBException(string(sqlite3_errmsg(_db)));
        }
    }

    Storage::~Storage() {
        sqlite3_close(_db);
    }

    void Storage::createSchema() {
        string sql = "CREATE TABLE actions ("  \
                         "id         INT PRIMARY KEY NOT NULL," \
                         "video_path TEXT NOT NULL," \
                         "type       TEXT NOT NULL);";
        int resCode = sqlite3_exec(_db, sql.c_str(), callback, nullptr, &_errMsg);
        if (resCode) {
            throw DBException(string(_errMsg));
        }
    }

    void Storage::insert(const Action &action) {
        string sql = "INSERT INTO actions(id, video_path, type)" \
                         "VALUES (" + std::to_string(action.id) + "," + action.videoPath + "," + action.type + ")";
        int resCode = sqlite3_exec(_db, sql.c_str(), callback, nullptr, &_errMsg);
        if (resCode) {
            throw DBException(string(_errMsg));
        }
    }

} // namespace detector
