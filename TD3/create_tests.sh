#!/bin/bash

# Create CSV file with header
echo "tissue,number_of_springs,contact" > tests.csv

# Generate combinations of tissue option, number of springs
for tissue in "spring" "None"; do
    if [ "$tissue" = "spring" ]; then
        options=("3" "5" "10")
    else
        options=("None")
    fi

    for num in "${options[@]}"; do
        for contact in "False" "True"; do
            echo "$tissue,$num,$contact" >> tests.csv
        done
    done
done

echo "CSV file 'tests.csv' created successfully!"
cat tests.csv