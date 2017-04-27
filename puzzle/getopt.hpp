#ifndef __GETOPT_H
#define __GETOPT_H

int getopt(int nargc, char * const nargv[], const char *ostr);

extern int optind;
extern char *optarg;

#endif
