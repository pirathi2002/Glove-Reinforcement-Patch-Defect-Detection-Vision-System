"""Create a GitHub repository via the GitHub API and optionally push the local repository.

Usage examples:

# Create repo under your user account (uses GITHUB_TOKEN env var or --token)
python scripts/create_github_repo.py --name glove_cv_project --private --description "Glove CV project"

# Create repo in an organization
python scripts/create_github_repo.py --name glove_cv_project --org my-org --private

# Create repo and push local files (automated):
python scripts/create_github_repo.py --name glove_cv_project --push --token YOUR_TOKEN_HERE --yes

Notes:
- Provide a personal access token with repo scopes via the GITHUB_TOKEN environment variable or --token argument.
- Pushing automatically will embed the token in the temporary remote URL; avoid using on shared systems if you are concerned about exposure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Optional

try:
    import requests
except Exception:
    print("The 'requests' package is required. Install: pip install requests")
    raise

API_BASE = "https://api.github.com"


def create_repo(token: str, name: str, private: bool = True, description: str = "", org: Optional[str] = None, auto_init: bool = False):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {
        "name": name,
        "private": private,
        "description": description,
        "auto_init": auto_init,
    }

    if org:
        url = f"{API_BASE}/orgs/{org}/repos"
    else:
        url = f"{API_BASE}/user/repos"

    resp = requests.post(url, headers=headers, json=payload)

    if resp.status_code in (201,):
        data = resp.json()
        print(f"Repository created: {data.get('html_url')}")
        return data
    else:
        print("Failed to create repository:")
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text)
        resp.raise_for_status()


def run_git_commands(repo_path: str, remote_url: str, token: Optional[str] = None, push: bool = True):
    def run(cmd, **kwargs):
        print(f"Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=repo_path, **kwargs)
        if res.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")

    # Init if needed
    if not os.path.isdir(os.path.join(repo_path, '.git')):
        run(['git', 'init'])
        run(['git', 'add', '.'])
        run(['git', 'commit', '-m', 'Initial commit'])

    # Add remote
    run(['git', 'remote', 'remove', 'origin'], check=False)

    run(['git', 'remote', 'add', 'origin', remote_url])
    # Set main branch
    run(['git', 'branch', '-M', 'main'])

    if push:
        # Push to remote
        run(['git', 'push', '-u', 'origin', 'main'])


def main():
    parser = argparse.ArgumentParser(description='Create GitHub repository via API and optionally push local project.')
    parser.add_argument('--name', required=True, help='Repository name')
    parser.add_argument('--private', action='store_true', default=False, help='Create repository as private (default: public)')
    parser.add_argument('--description', default='', help='Repository description')
    parser.add_argument('--org', default=None, help='Organization name (create repo under an org)')
    parser.add_argument('--token', default=None, help='GitHub personal access token (or use GITHUB_TOKEN env var)')
    parser.add_argument('--auto-init', action='store_true', help='Auto initialize repository on GitHub (add README)')
    parser.add_argument('--push', action='store_true', help='Attempt to push local repository to the created GitHub repo')
    parser.add_argument('--repo-path', default='.', help='Local repository path to push (default: current dir)')
    parser.add_argument('--yes', action='store_true', help='Proceed without asking for confirmation (use with caution)')

    args = parser.parse_args()

    token = args.token or os.environ.get('GITHUB_TOKEN')
    if not token:
        print('Error: GitHub token required. Provide via --token or GITHUB_TOKEN environment variable.')
        sys.exit(1)

    # Create repo
    data = create_repo(token=token, name=args.name, private=args.private, description=args.description, org=args.org, auto_init=args.auto_init)

    owner = data.get('owner', {}).get('login')
    if not owner:
        print('Could not determine repository owner from API response. Aborting push step if requested.')

    html_url = data.get('html_url')
    clone_url = data.get('clone_url')

    if args.push:
        if not args.yes:
            confirm = input(f"Repository {html_url} created. Proceed to push local '{args.repo_path}' to this remote? [y/N]: ")
            if confirm.lower() != 'y':
                print('Push aborted by user.')
                return

        # Construct remote URL with token for non-interactive push (HTTPS)
        # NOTE: embedding tokens in URLs can be exposed in process lists—use with caution.
        remote_with_token = clone_url.replace('https://', f'https://{token}@')

        try:
            run_git_commands(repo_path=os.path.abspath(args.repo_path), remote_url=remote_with_token, token=token, push=True)
            print('Push completed successfully.')
        except Exception as exc:
            print('Push failed:', exc)
            print('You can add remote and push manually:')
            print(f'  git remote add origin {clone_url}')
            print('  git branch -M main')
            print('  git push -u origin main')


if __name__ == '__main__':
    main()
