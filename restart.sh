#!/usr/bin/env bash

set +e

while true; do
    echo "===== START $(date) ====="

    python pipeline.py \
        --config config \
        --experiment catalan_plus_usatges_upper_triangle

    EXIT_CODE=$?

    echo "===== END $(date) | exit code: $EXIT_CODE ====="

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "SUCCESS — exiting loop"
        break
    fi

    echo "FAILED — restarting in 5 seconds..."
    sleep 5
done
