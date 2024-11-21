import git

# URL of the repository to clone
repo_url = "https://github.com/Misakalastorder/UWImage-Restoration.git"
# Local directory to clone the repository into
local_dir = "D:\\2024\\pccup\\big\\clone"

# Clone the repository
git.Repo.clone_from(repo_url, local_dir)