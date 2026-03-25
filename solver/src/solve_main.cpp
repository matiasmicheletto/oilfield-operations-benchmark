#define MANUAL "assets/solve_manual.txt"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <getopt.h>

#include "../include/utils.h"


static struct option long_options[] = {
    {"help",        no_argument,        0,  'h' },
    {"version",     no_argument,        0,  'v' },
    {"config",      required_argument,  0,  'c' },
    {"set",         required_argument,  0,  's' },
    {"output",      required_argument,  0,  'o' },
    {"dbg",         no_argument,        0,  'd' },
    {0,             0,                  0,  0   }
};

int main(int argc, char **argv) {

    // Parse command line arguments
    int opt;
    int option_index = 0;
    std::string cfg_filename = "config.yaml"; // Solver config file (yaml)
    std::vector<std::string> cfg_overrides; // Configuration overrides from command line

    while((opt = getopt_long(argc, argv, "vhd:c:o:s", long_options, &option_index)) != -1) {
        switch(opt) {
            case 'v': // version
                std::cout << "Solver version 1.0.0" << std::endl;
                return 0;
            case 'h': // help
                utils::printHelp(MANUAL);
                return 0;
            case 'c': // config
                cfg_filename = optarg;
                break;
            case 's': // set config override
                cfg_overrides.emplace_back(optarg);
                break;
            case 'o': // output file name
                break;
            case 'd': // enable debug output
                utils::dbg.rdbuf(std::cout.rdbuf());
                break;
            case '?': // unknown option
                return 1;
        }
    }

    return 0;
}

