# CEEP v1.0.0 Release Checklist

## Pre-Release

- [x] Rename neurowave → ceep
- [x] Update README with professional content
- [x] Create AUTHORS.md with proper credits
- [x] Create CITATION.cff
- [x] Update pyproject.toml metadata
- [ ] Run full test suite: `pytest tests/`
- [ ] Check all examples run: `python examples/*.py`
- [ ] Verify GPU code works on T4/V100/A100

## Documentation

- [ ] Complete API documentation
- [ ] Add Jupyter notebook tutorials
- [ ] Create video demo (YouTube)
- [ ] Write blog post announcement

## Repository

- [ ] Add project logo (docs/assets/ceep_logo.png)
- [ ] Create GitHub Actions CI/CD
- [ ] Setup Read the Docs
- [ ] Add issue templates
- [ ] Create pull request template

## Distribution

- [ ] Build package: `python -m build`
- [ ] Test install: `pip install dist/ceep-1.0.0.tar.gz`
- [ ] Upload to Test PyPI
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Create GitHub release v1.0.0
- [ ] Tag release: `git tag v1.0.0`

## Announcement

- [ ] Post on GitHub Discussions
- [ ] Tweet announcement
- [ ] Post on Reddit (r/MachineLearning, r/CUDA)
- [ ] Email biomedical imaging labs
- [ ] Submit to Papers with Code

## Paper Submission

- [ ] Finalize IEEE ISBI 2026 paper
- [ ] Generate all figures
- [ ] Run final benchmarks
- [ ] Submit before deadline (Nov 2025)

---

*Created: 2026-05-14*
*By: Shahzaib Ur Rehman*
