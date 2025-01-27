#!/bin/sh
if [ "$1" = "-" ]; then
    # Read from stdin
    cat > /tmp/program.cpp
    input_file=/tmp/program.cpp
else
    input_file="$1"
fi

g++ "$input_file" -o /tmp/program
if [ $? -eq 0 ]; then
    /tmp/program
else
    echo "Compilation failed"
    exit 1
fi