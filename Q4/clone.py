import git
import os

# URL of the repository to clone
repo_url = "https://github.com/saeed-anwar/UWCNN.git"

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Local directory to clone the repository into
local_dir = os.path.join(script_dir, "clone")

# Clone the repository
git.Repo.clone_from(repo_url, local_dir)