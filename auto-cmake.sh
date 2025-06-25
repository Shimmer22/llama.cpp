#!/bin/bash
cmake -B build -DLLAMA_CURL=OFF "$@"
cmake --build build --config Release -j 8