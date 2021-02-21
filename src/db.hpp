#include <sqlite3.h>

#include <utility>

#include "multitracker.hpp"

namespace detector {


    struct Action {
        int id;
        string videoPath;
        string type;
    };

    int callback(void *, int, char **, char **);

    class DBException : public std::exception {
    private:

        string _errMessage;

    public:

        explicit DBException(string);

        virtual const char *what() noexcept;

    };

    class Storage {
    public:

        sqlite3 *_db;
        char *_errMsg;

    public:

        explicit Storage(const string &);

        ~Storage();

        void createSchema();

        void insert(const Action &action);

    };

} // namespace detector
