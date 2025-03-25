#!/bin/bash
# Script: push_to_github.sh
# Description: Adds all changes, commits with a timestamped message, and pushes to GitHub.
# Usage: Place this script in your repository root, make it executable (chmod +x push_to_github.sh), then run it.

# Variables - adjust these as needed:
REMOTE_NAME="origin"
BRANCH_NAME="main"
COMMIT_MESSAGE="Auto-update from VM on $(date +"%Y-%m-%d %H:%M:%S")"

# Check if current directory is a Git repository
if [ ! -d ".git" ]; then
    echo "Error: This directory is not a Git repository."
    exit 1
fi

# Stage all changes
git add .

# Check if there are any changes to commit
if git diff-index --quiet HEAD --; then
    echo "No changes detected. Nothing to commit."
else
    # Commit changes
    git commit -m "$COMMIT_MESSAGE"
fi

# Push changes to the specified remote and branch
echo "Pushing changes to ${REMOTE_NAME}/${BRANCH_NAME}..."
git push $REMOTE_NAME $BRANCH_NAME

echo "Push completed."
