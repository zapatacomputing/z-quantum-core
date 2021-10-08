These pipelines require two variables at set-pipeline.  The other variables are provided by Hashicorp Vault

github_branch
docker_tag

concourse's documentation can be found here: https://concourse-ci.org but the quick version for variables is to use the syntax `-v KEY=VALUE` in the set-pipeline command.