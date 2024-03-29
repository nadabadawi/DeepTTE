#!/bin/bash

# Update package lists
sudo apt-get update

# Install packages listed in requirements.txt
while read -r package; do
    sudo apt-get install -y "$package"
done < dependencies.txt

echo "All dependencies installed successfully."
