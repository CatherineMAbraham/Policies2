#!/bin/bash

# Create CSV file with header
> tests.csv

# Generate combinations of tissue option, number of springs
for tissue in "spring"; do
    if [ "$tissue" = "spring" ]; then
        options=("1" "3" "5" "10")
    else
        options=("1")
    fi

    for num in "${options[@]}"; do
        for contact in "0"; do
            echo "$tissue,$num,$contact" >> tests.csv
        done
    done
done

echo "CSV file 'tests.csv' created successfully!"
cat tests.csv