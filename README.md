# C++ Quiz Code Runner

A tool for generating and testing "dry runs" (tricky C++ code snippets often asked in exams by universities like FAST) in an isolated Docker environment.

## Overview

This project provides a system for:
- Generating challenging C++ code snippets that test understanding of tricky language concepts
- Running the code snippets safely in a containerized environment 
- Logging the results for verification

## Features

- Focuses on tricky C++ concepts like:
  - sizeof() and datatype size edge cases
  - Order of operations (e.g., x++ + x++)
  - Conditional statement edge cases
  - Loop patterns
  - Pointer manipulation and const correctness

- Safe execution environment using Docker
- Detailed logging of code snippets and their outputs
- Automated testing and verification

## Setup

1. Prerequisites:
   - Python 3.x
   - Docker
   - Required Python packages (see requirements)

2. Environment Variables:
   - Create a `.env` file with:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## Usage

Run the main script:

```sh
python main.py
```