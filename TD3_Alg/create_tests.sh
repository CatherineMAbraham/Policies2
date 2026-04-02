#!/bin/bash

# Create CSV file with header
> tests.csv

# Generate combinations of tissue option, number of springs
# for tissue in "spring" "None"; do
#     if [ "$tissue" = "spring" ]; then
#         options=("1" "3" "5" "10")
#         ymoptions=("1e6" "5e6" "1e7")
#     else
#         options=("1")
#         ymoptions=("1")
#     fi

#     for num in "${options[@]}"; do
#         for ym in "${ymoptions[@]}"; do
#             for seed in "1" "2" "3" "4" "5" "6"; do
#                 echo "$tissue,$num,$ym,$seed" >> tests.csv
#             done
#         done
#     done
# done
for tissue in "spring" "None"; do
    if [ "$tissue" = "spring" ]; then
        options=("1" "3" "5" "10")
        ymoptions=("1e6" "5e6" "1e7")
    else
        options=("1")
        ymoptions=("1")
    fi

    for num in "${options[@]}"; do
        for ym in "${ymoptions[@]}"; do
            for seed in "1"; do
                echo "$tissue,$num,$ym,$seed" >> tests.csv
            done
        done
    done
done
echo "CSV file 'tests.csv' created successfully!"
cat tests.csv