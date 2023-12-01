import os.path

import git

CommitHash = str
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_current_commit() -> CommitHash:
    repo = git.Repo(REPO_PATH)
    sha = repo.head.object.hexsha
    return sha
