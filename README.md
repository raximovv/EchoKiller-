# Day 16 — EchoKiller

EchoKiller is a Python project that reduces echo using an adaptive FIR filter.

## Features

- Loads a `.wav` audio file or generates a synthetic sample
- Adds echo to the signal
- Applies an adaptive FIR filter to reduce echo
- Shows before/after waveforms
- Visualizes learned filter coefficients
- Saves `echoed_input.wav` and `cleaned_output.wav`

## Tech Stack

- numpy
- scipy
- soundfile
- matplotlib
- sounddevice
