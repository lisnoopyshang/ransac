#include <sys/time.h>
#include <iostream>
#include <map>
#include <string>

using namespace std;

struct Timer {

    map<string, struct timeval> startTime;
    map<string, struct timeval> stopTime;
    map<string, double>         time;

    void start(string name) {
        if(!time.count(name)) {
            time[name] = 0.0;
        }
        gettimeofday(&startTime[name], NULL);
    }

    void stop(string name) {
        gettimeofday(&stopTime[name], NULL);
        time[name] += (stopTime[name].tv_sec - startTime[name].tv_sec) * 1000000.0 +
                      (stopTime[name].tv_usec - startTime[name].tv_usec);
    }

	void print(string name, int REP) { printf("%s Time (ms): %f\n", name.c_str(), time[name] / (1000 * REP)); }//ms
};