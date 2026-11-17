# Contributing

Thanks for your interest in improving this project! Bug reports, ideas and pull requests are welcome.

## Reporting bugs

Open an issue and include:

- Your operating system and Python version (`python --version`)
- How you installed `dlib` / `face_recognition`
- The exact command you ran and the full error output
- Your `config.json` (remove anything private, such as custom font paths with your username)

**Please never attach photos of real people to public issues** unless you have their explicit permission.

## Suggesting changes

For larger changes, open an issue first to discuss the idea. The project aims to stay small and easy to read, so changes that add heavy dependencies or large refactors are unlikely to be accepted.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep the change focused: one fix or feature per pull request.
3. Preserve existing behavior unless the change is the point of the pull request.
4. Test manually with a real camera and a few reference photos, and describe what you tested on which OS.
5. Update `README.md` and `CHANGELOG.md` if user-facing behavior changes.

## Code style

- Follow the style already used in `face_system.py` (PEP 8, short functions, comments only where they help).
- Do not commit photos, logs, cache files or `config.json`. These are excluded by `.gitignore`; please keep it that way.
