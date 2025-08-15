#include <string>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <cmath>  
#include <numeric> 
#include <random>
#include <sstream>
#include <stdio.h>
#include <tuple>
#include <vector>
using namespace std;


string file_name = "results/Aug_15_1_rep_10_max_iter_1000_conv_thresh_0.0005.diri_prob";

std::ifstream inputFile(file_name);

inline vector<string> split_ws(const string& s) {
    istringstream iss(s);
    vector <string> out;
    for (string tok; iss >> tok;) out.push_back(tok);
    return out;
}

static inline void dump_bytes(const std::string& s) {
    std::cerr << "  bytes:";
    for (unsigned char ch : s) std::cerr << " " << std::hex << std::uppercase
                                         << std::setw(2) << std::setfill('0') << (int)ch;
    std::cerr << std::dec << "\n";
}

int main () {    

    std::string line;
    // skip first two header lines
    std::string node_name;
    // string node_parent_name;
    double prob; int i, j;
    std::getline(inputFile, line);
    std::getline(inputFile, line);
    while (std::getline(inputFile, line)) {
        vector<string> splitLine = split_ws(line);
        const int num_words = static_cast<int>(splitLine.size());
        // cout << num_words << endl;
        switch (num_words) {
            case 8: { 
                // node_parent_name = splitLine[5];
                node_name = splitLine[7];
                break;
            }
            case 16: {                
                for (int p_id = 0; p_id < 16; ++p_id) {
                    i = p_id / 4;
                    j = p_id % 4;
                    try
                    {
                        prob = stod(splitLine[p_id]);
                        // cout << "parsed probability " << prob << endl;  
                        // dump_bytes(splitLine[p_id]);                      				
                    }
                    catch(const exception& e)
                    {						
                        // cout << splitLine[p_id] << '\n';
                        cout << "probability not parsed " << splitLine[p_id]  << endl;
                        // dump_bytes(splitLine[p_id]);
                        // throw mt_error("string not converted to double");
                        
                    }
                    
                }
                break;
            }
            case 9: {                
                node_name = splitLine[3];                
                break;
            }
            case 4: {                
                for (int p_id = 0; p_id < 4; ++p_id) {
                    prob = std::stod(splitLine[p_id]);                    
                }
                break;
            }
            default:
                std::cerr << "ReadProbabilities: unexpected token count (" << num_words
                            << ") on line: " << line << "\n";
                break;
        }
    }
}

