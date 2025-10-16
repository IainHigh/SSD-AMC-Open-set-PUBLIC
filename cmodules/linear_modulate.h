// 
// linear_modulate.h
//

#ifndef LINEAR_MODULATE_H
#define LINEAR_MODULATE_H

void linear_modulate(int modtype, int order, int n_sym, unsigned int s[], float rI[], float rQ[], int verbose, int seed);
void linear_constellation(int modtype, int order, float rI[], float rQ[]);

#endif
