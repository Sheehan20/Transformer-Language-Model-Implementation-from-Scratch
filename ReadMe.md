## Overview

This repository contains my coursework implementations for the Stanford CS336 (LLM From Scratch) class. The goal is to build a simplified Transformer language model from scratch and explore system-level optimizations.

Main directories:

- `assignment1-basics`: Core model building blocks and unit tests
  - Simple BPE tokenizer
  - Rotary positional embeddings (RoPE)
  - Multi-head self-attention and feed-forward network
  - Optimizer (e.g., AdamW)
  - A minimal Transformer language model
- `assignment2-systems`: System acceleration and parallel training experiments


## Quick Start

1. Clone this repository and change into the project root.
2. Navigate into a specific assignment directory (e.g., `assignment1-basics`).
3. Follow the instructions in that directory to install dependencies and run tests (e.g., with `pytest`).

Tip: Each assignment directory provides a `pyproject.toml` and test cases. It is recommended to use a virtual environment.

## Repository Structure

```
assets/                             # Images and auxiliary assets
assignment1-basics/                 # Assignment 1: core implementations and tests
assignment2-systems/                # Assignment 2: system and parallelism work
ReadMe.md                           # This file
```
