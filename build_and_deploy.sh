#!/bin/bash
# Navigate to the project directory
cd /home/team21/Documents/group-project-s24-caped-crusaders

# Generate a tag using the current timestamp
TAG=$(date +%m%d%H)

# Build the Docker image with the generated tag
docker build -t team21:"$TAG" .

# Optional: Output the name and tag of the built image
echo "Built image: team21:$TAG"

# Force update the Docker Swarm service
docker service update --force --image team21:"$TAG" team21stack_app

