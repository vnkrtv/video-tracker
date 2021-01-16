#include "argparse.hpp"

int main(int argc, char const *argv[]) {
    args::parse<detector::Args>(argc, argv);
}
