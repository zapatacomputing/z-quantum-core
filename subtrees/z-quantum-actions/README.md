# z-quantum-actions

This repo contains definitions of custom GitHub Action used in `z-quantum-*` and `qe-*`
repositories.

## Usage

This repo is private, so actions can't be referenced directly via `uses:` [step keyword](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#jobsjob_idstepsuses).
The recommended workaround is to add this repo into your "client" repo using git subtree:

```bash
cd your-client-repo
git subtree add -P vendor/z-quantum-actions git@github.com:zapatacomputing/z-quantum-actions.git master --squash
```

Actions can be then referred to via their local paths, like:
```yaml
# inside your GitHub workflow
steps:
  - name: Run publish release action
    uses: ./vendor/z-quantum-actions/publish-release
```

## Repo contents

- `publish-release` - single action that wraps a couple of workflow steps
- `workflow-templates` - workflow .ymls that can't be used directly, but can be copied to your repo to set up the release process.
