#!/bin/bash

# Create CSV file with header
> tests.csv

# Generate combinations of tissue option, number of springs
for tissue in "spring" None; do
    if [ "$tissue" = "spring"  ]; then
        options=("1e7" "5e6" "1e6")
    else
        options=("1")
    fi

    for ym in "${options[@]}"; do
        for max_force in "3" "3.5"; do
            echo "$tissue,$ym,$max_force" >> tests.csv
        done
    done
done

echo "CSV file 'tests.csv' created successfully!"
cat tests.csv