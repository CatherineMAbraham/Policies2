#!/bin/bash

# Create CSV file with header
> tests.csv

# Generate combinations of tissue option, number of springs
for tissue in "spring"; do
    for ym in "1e6" "1e7" "5e6"; do
        for num_springs in "3" "5" "10"; do
            echo "$tissue,$ym,$num_springs" >> tests.csv
        done
    done
done

echo "CSV file 'tests.csv' created successfully!"
cat tests.csv