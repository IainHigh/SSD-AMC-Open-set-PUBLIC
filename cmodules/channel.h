// 
// channel.h
//

#ifndef CHANNEL_H
#define CHANNEL_H

void channel(float snr, int n_sym, int sps, float fo, float po, float xI[], float xQ[], float yI[], float yQ[], int verbose, int seed);
void rayleigh_channel(float snr, int n_sym, int sps, float fo, float po, int num_taps, int awgn, 
                    float xI[], float xQ[], float yI[], float yQ[], float path_delays[], float path_gains[], int verbose, int seed);
void rician_channel(float snr, int n_sym, int sps, float fo, float po, float k_factor, int num_taps, int awgn, 
                    float xI[], float xQ[], float yI[], float yQ[], float path_delays[], float path_gains[], int verbose, int seed);


#endif
