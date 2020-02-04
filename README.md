# FFT1d for FPGA

This repository contains OpenCL implementation of FFT1d provided by Intel Design Samples. It has been modified to enable hyperflex optimization for Intel Stratix 10 FPGAs. In order to do this, the fetch kernel in the `fft1d.cl` file has been refactored as a `SingleWorkItem` kernel. This incidentally also enables FFT1d of sizes lower than 2<sup>6</sup> and larger than 2<sup>12</sup>.

## Build

This steps comprises of two phases, build the host code and the kernel code. Both are dependent on the file `host/inc/fft_config.h`, which sets the size of the FFT1d to be synthesized. The size is set as a log of its value.

### Build host

    cd fft1d
    make

### Emulation

Using legacy-emulator

    aoc -g -v -march=emulator -legacy-emulator -no-interleaving=default -fpc fft1d/device/fft1d.cl -o fft1d/bin/fft1d.aocx

### Report

    aoc -g -v -report -rtl -no-interleaving=default -fpc fft1d/device/fft1d.cl -o fft1d/report/fft1d.aocr

### Synthesis

    aoc -v -fpc -no-interleaving=default -board=p520_hpc_sg280l fft1d/device/fft1d.cl -o fft1d/bin/fft1d.aocx 

## Execution

Emulation

    env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 bin/host

Synthesized

    bin/host

## Output

The output verifies the result of the FPGA execution with a CPU FFT implementation. The output of the implementation is in bit-reversed order.

## Performance Comparison

Compare performance of the FPGA execution with the model

### Latency

| FFT Size | Expected Latency (microsec)  |   Latency Reference (microsec)  | Latency with Hyperflex (microsec)
|:--------:|:------------------:|:------------------:|:--------------:|
|    32    |      0.013         |                    |    0.271       |
|    64    |      0.027         |     0.032          |    0.246       |
|    128   |      0.053         |    0.063           |    0.260       |
|    256   |      0.107         |    0.126           |    0.303       |

The empty value for the size 32 of the reference implementation is because the code cannot be synthesized out-of-the-box for FFT values below 64 and needs refactoring.

All measurements have been performed as the following:

- Latency measured as an average of 100000 iterations for every given FFT size.
- Kernel code is synthesized using Intel OpenCL SDK version 19.3 and the Nallatech BSP version 19.2.0_hpc.
- Host code is compiled using gcc 8.3.0.

### Kernel frequencies 

| FFT Size |    Reference FFT  (Mhz) |  FFT with Hyperflex (Mhz) |
|:--------:|:-----------------:|:-------------------------:|
|    32    |                   |       331.67              |
|    64    |       343.87      |       340.83              |
|    128   |       358.29      |       371.47              |
|    256   |       357.39      |       336.24              |

## Analysis

1. Latency of the bitstreams with hyperflex enabled did not improve the latency of the FFT1d execution instead made it worse by factors 3 - 8 as shown in the table. This shows that the Single Work-Item Fetch kernel needs to be optimized further to offer similar performance as the NDRange Fetch Kernel.

2. Even though hyperflex is enabled the clock frequencies are not drastically improved, if at all.