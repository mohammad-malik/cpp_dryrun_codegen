# Agentic C++ Dry Run Code Generator and Runner

A tool for generating and testing "dry runs" (tricky C++ code snippets often asked in exams by universities like FAST) in an isolated Docker environment.

## Overview

This project provides a system for:
- Generating challenging C++ code snippets that test understanding of tricky language concepts taking inspiration from various PF and OOP past papers through Retrieval Augmented Generation (RAG).
- Running the code snippets safely in a virtual execution environment using Docker, to ensure accuracy.
- Automated testing and verification (as the outputs are fed back to a reasoning LLM, which generates until it has 10 various questions that meet the criteria and are worth studying).
- A web interface to view the newly generated code snippets, their outputs, and an explanation for the output.

## Features

- Focuses on tricky C++ concepts like:
  - sizeof() and datatype size edge cases
  - Order of operations (e.g., x++ + x++)
  - Conditional statement edge cases
  - Loop patterns
  - Pointer manipulation and const correctness
  among many others asked by FAST in courses like Programming Fundamentals (PF) and Object Oriented Programming (OOP).

## Setup

1. Prerequisites:
   - Python 3.x
   - Docker
   - Required Python packages (see requirements)

2. Environment Variables:
   - Generate an API Key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file with:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```  

## Usage

### With python virtual env:
1. Download `install.sh`
2. Set it as executable: `chmod +x install.sh`
3. Wait.
4. When done, set `run.sh` as executable: `chmod +x ./run.sh`
5. Run it. `./run.sh`

### Alternatively, for those with conda:

1. Download `install_condabased.sh`
2. Follow the steps above.

