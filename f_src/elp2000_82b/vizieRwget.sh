#!/bin/bash

# Base URL
base_url="https://cdsarc.cds.unistra.fr/ftp/VI/79/"

# Loop through the file numbers and download each one
for i in {1..36}
do
    wget "${base_url}ELP${i}"
done