// Copyright (C) 2013-2019 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

// Modified by Arjun Ramaswami, arjun.ramaswami@upb.de

/* This is the top-level device source file for the fft1d example. The code is
 * written as an OpenCL single work-item kernel. This coding style allows the 
 * compiler to extract loop-level parallelism from the source code and 
 * instantiate a hardware pipeline capable of executing concurrently a large 
 * number of loop iterations. The compiler analyses loop-carried dependencies, 
 * and these translate into data transfers across concurrently executed loop 
 * iterations. 
 *
 * Careful coding ensures that all loop-carried dependencies are trivial, 
 * merely data transfers which span a single clock cycle. The FFT algorithm 
 * requires passing data forward across loop iterations. The code uses a 
 * sliding window to implement these data transfers. The advantage of using a
 * sliding window is that dependencies across consecutive loop iterations have
 * an invariant source and destination (pairs of constant offset array 
 * elements). Such transfer patterns can be implemented efficiently by the 
 * FPGA hardware. All this ensures an overall processing a throughput of one 
 * loop iteration per clock cycle.
 *
 * The size of the FFT transform can be customized via an argument to the FFT 
 * engine. This argument has to be a compile time constant to ensure that the 
 * compiler can propagate it throughout the function body and generate 
 * efficient hardware.
 */

// Include source code for an engine that produces 8 points each step
#include "fft_8.cl" 

#pragma OPENCL EXTENSION cl_intel_channels : enable

#include "../host/inc/fft_config.h"

#define min(a,b) (a<b?a:b)

#define LOGPOINTS       3
#define POINTS          (1 << LOGPOINTS)

// Need some depth to our channels to accommodate their bursty filling.
//channel float2 chanin[8] __attribute__((depth(CONT_FACTOR*8)));
channel float2 chanin[8] __attribute__((depth((1<<LOGN)*8)));

uint bit_reversed(uint x, uint bits) {
  uint y = 0;
  #pragma unroll 
  for (uint i = 0; i < bits; i++) {
    y <<= 1;
    y |= x & 1;
    x >>= 1;
  }
  y &= ((1 << bits) - 1);
  return y;
}

__kernel
__attribute__ ((max_global_work_dim(0)))
void fetch(global volatile float2 * restrict src, int iter) {

  const int N = (1 << LOGN);
  const int BUF_SIZE = N;

  for(unsigned k = 0; k < iter; k++){ 

    float2 buf[BUF_SIZE];
    #pragma unroll 8
    for(int i = 0; i < N; i++){
      buf[i & ((1<<LOGN)-1)] = src[(k << LOGN) + i];    
    }

    for(unsigned j = 0; j < (N / 8); j++){
      write_channel_intel(chanin[0], buf[j]);               // 0
      write_channel_intel(chanin[1], buf[4 * N / 8 + j]);   // 32
      write_channel_intel(chanin[2], buf[2 * N / 8 + j]);   // 16
      write_channel_intel(chanin[3], buf[6 * N / 8 + j]);   // 48
      write_channel_intel(chanin[4], buf[N / 8 + j]);       // 8
      write_channel_intel(chanin[5], buf[5 * N / 8 + j]);   // 40
      write_channel_intel(chanin[6], buf[3 * N / 8 + j]);   // 24
      write_channel_intel(chanin[7], buf[7 * N / 8 + j]);   // 54
    }
  }
}

/* Attaching the attribute 'task' to the top level kernel to indicate 
 * that the host enqueues a task (a single work-item kernel)
 *
 * 'src' and 'dest' point to the input and output buffers in global memory; 
 * using restrict pointers as there are no dependencies between the buffers
 * 'count' represents the number of 4k sets to process
 * 'inverse' toggles between the direct and the inverse transform
 */

__attribute((task))
kernel void fft1d(global float2 * restrict dest,
                  int count, int inverse) {

  const int N = (1 << LOGN);

  /* The FFT engine requires a sliding window array for data reordering; data 
   * stored in this array is carried across loop iterations and shifted by one 
   * element every iteration; all loop dependencies derived from the uses of 
   * this array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + 8 * (LOGN - 2)];

  /* This is the main loop. It runs 'count' back-to-back FFT transforms
   * In addition to the 'count * (N / 8)' iterations, it runs 'N / 8 - 1'
   * additional iterations to drain the last outputs 
   * (see comments attached to the FFT engine)
   *
   * The compiler leverages pipeline parallelism by overlapping the 
   * iterations of this loop - launching one iteration every clock cycle
   */

  for (unsigned i = 0; i < count * (N / 8) + N / 8 - 1; i++) {

    /* As required by the FFT engine, gather input data from 8 distinct 
     * segments of the input buffer; for simplicity, this implementation 
     * does not attempt to coalesce memory accesses and this leads to 
     * higher resource utilization (see the fft2d example for advanced 
     * memory access techniques)
     */

    int base = (i / (N / 8)) * N;
    int offset = i % (N / 8);

    float2x8 data;
    // Perform memory transfers only when reading data in range
    if (i < count * (N / 8)) {
      data.i0 = read_channel_intel(chanin[0]);
      data.i1 = read_channel_intel(chanin[1]);
      data.i2 = read_channel_intel(chanin[2]);
      data.i3 = read_channel_intel(chanin[3]);
      data.i4 = read_channel_intel(chanin[4]);
      data.i5 = read_channel_intel(chanin[5]);
      data.i6 = read_channel_intel(chanin[6]);
      data.i7 = read_channel_intel(chanin[7]);
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    // Perform one step of the FFT engine
    data = fft_step(data, i % (N / 8), fft_delay_elements, inverse, LOGN); 

    /* Store data back to memory. FFT engine outputs are delayed by 
     * N / 8 - 1 steps, hence gate writes accordingly
     */

    if (i >= N / 8 - 1) {
      int base = 8 * (i - (N / 8 - 1));
 
      // These consecutive accesses will be coalesced by the compiler
      dest[base] = data.i0;
      dest[base + 1] = data.i1;
      dest[base + 2] = data.i2;
      dest[base + 3] = data.i3;
      dest[base + 4] = data.i4;
      dest[base + 5] = data.i5;
      dest[base + 6] = data.i6;
      dest[base + 7] = data.i7;
    }
  }
}

