#include "log.hpp"
#include <ctime>

using namespace std;

static clock_t _last_clock;

void log(string msg) {
	clog << msg << endl;
}

void logd(string msg, bool display_time) {
#ifdef _DEBUG
	if (display_time) {
		clock_t elasped_clock = clock() - _last_clock;
		clog << msg << ". Elasped time is " << (float)elasped_clock / CLOCKS_PER_SEC << "sec." << endl;
	}
	else {
		clog << msg << endl;
	}
	_last_clock = clock();
#endif // _DEBUG
}