# Developing the DeepSparse Engine

[TODO: ENGINEERING - ADJUST SECTION HEADERS AS RELEVANT TO SOFTWARE; REMOVE THESE TODO(S) AND EXAMPLES BELOW]

[ENGINEERING: State OS/Environment reminder; requirements upfront]

Here's some details to get started.

## Basic Commands

[TODO: ENGINEERING]

**EXAMPLE: install commands?**

```bash
[ENGINEERING: INSERT HERE]
```

This will do [ENGINEERING: list out concise summary of what these command will do and what the use can expect to happen once installed]..

**EXAMPLE: makefile commands?**

```bash
[ENGINEERING: INSERT HERE]
```

This will do [ENGINEERING: list out concise summary of what these command will do and what the use can expect to happen once installed].

**EXAMPLE: test changes locally?**

```bash
[ENGINEERING: INSERT HERE]
```

This will do [ENGINEERING: list out concise summary of what these command will do and what the use can expect to happen once installed]..

## Resources

[TODO: ENGINEERING - What does the developer need to know?]

- EXAMPLE: architecture docs?
- EXAMPLE: how make a modifier? (separate doc)

## GitHub Workflow

1. Fork the `neuralmagic/deepsparse` repository into your GitHub account: https://github.com/neuralmagic/deepsparse/fork.

2. Clone your fork of the GitHub repository, replacing `<username>` with your GitHub username.

   Use ssh (recommended):

   ```bash
   git clone git@github.com:<username>/deepsparse.git
   ```

   Or https:

   ```bash
   git clone https://github.com/<username>/deepsparse.git
   ```

3. Add a remote to keep up with upstream changes.

   ```bash
   git remote add upstream https://github.com/neuralmagic/deepsparse.git
   ```

   If you already have a copy, fetch upstream changes.

   ```bash
   git fetch upstream
   ```

4. Create a feature branch to work in.

   ```bash
   git checkout -b feature-xxx remotes/upstream/master
   ```

5. Work in your feature branch.

   ```bash
   git commit -a
   ```

6. Periodically rebase your changes

   ```bash
   git pull --rebase
   ```

7. When done, combine ("squash") related commits into a single one

   ```bash
   git rebase -i upstream/master
   ```

   This will open your editor and allow you to re-order commits and merge them:
   - Re-order the lines to change commit order (to the extent possible without creating conflicts)
   - Prefix commits using `s` (squash) or `f` (fixup) to merge extraneous commits.

8. Submit a pull-request

   ```bash
   git push origin feature-xxx
   ```

   Go to your fork main page

   ```bash
   https://github.com/<username>/deepsparse
   ```

   If you recently pushed your changes GitHub will automatically pop up a `Compare & pull request` button for any branches you recently pushed to. If you click that button it will automatically offer you to submit your pull-request to the `neuralmagic/deepsparse` repository.

   - Give your pull-request a meaningful title.
     You'll know your title is properly formatted once the `Semantic Pull Request` GitHub check
     transitions from a status of "pending" to "passed".
   - In the description, explain your changes and the problem they are solving.

9. Addressing code review comments

   Repeat steps 5. through 7. to address any code review comments and rebase your changes if necessary.

   Push your updated changes to update the pull request

   ```bash
   git push origin [--force] feature-xxx
   ```

   `--force` may be necessary to overwrite your existing pull request in case your
  commit history was changed when performing the rebase.

   Note: Be careful when using `--force` since you may lose data if you are not careful.

   ```bash
   git push origin --force feature-xxx
   ```
