import os

# Define the repository URL
repo_url = "https://github.com/Misakalastorder/UWImage-Restoration.git"

# Initialize a new Git repository
os.system("git init")

# Add all files to the repository
os.system("git add .")

# Commit the changes
os.system('git commit -m "Initial commit"')

# Add the remote repository URL
os.system(f"git remote add origin {repo_url}")

# Pull the changes from the remote repository
os.system("git pull origin master --rebase")

# Push the changes to the remote repository
os.system("git push -u origin master")
# # Push the changes to the remote repository on the 'second' branch
# os.system("git push -u origin second")