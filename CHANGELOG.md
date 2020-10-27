# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
### Fixed
### Removed

## [0.9.1] - 2020-10-27
### Fixed
- Fixed the conda build for pytorch versions < 1.5

## [0.9.0] - 2020-10-26
### Added
- Support for Python 3.8
- Suport for PyTorch 1.5
### Fixed
- Fixed normalization for multi-channel input and output.
### Removed
- Support for PyTorch 1.1

## [0.8.0] - 2020-03.10
### Added
### Fixed
- Weights access when pruning in pytorch 1.4
### Removed
- torchvision dependency
- sacred dependency
- `msd_pytorch.relu_inplace`
- command-line interface
- old MSDModule
- stitch functions and modules

## [0.7.3] - 2020-03-10
### Fixed
- Bug in relabeling code in ImageDataset.

## [0.7.2] - 2019-07-30
### Added
- Support for multi-gpu execution. Use `parallel=True` when
  constructing a `MSDRegressionModel` or `MSDSegmentationModel`.
### Fixed
- Make `model.forward()` more memory-efficient.
### Removed

## [0.7.1] - 2019-05-27
### Added
- Add `weights_path` command-line argument to msd executable to indicate
  where to store final weights.
- Add `MSDBlock2d`: this is a faster and slightly more memory efficient
  implementation of the same MSD component. Many thanks to Jonas
  Adler for suggesting this way of structuring the code!
### Changed
- The MSD models use `MSDBlock2d` implementation by default now.

## 0.6.2 - 2019-05-23
### Added
- Initial release.

[Unreleased]: https://www.github.com/ahendriksen/msd_pytorch/compare/v0.9.1...master
[0.9.1]: https://www.github.com/ahendriksen/msd_pytorch/compare/v0.9.0...v0.9.1
[0.9.0]: https://www.github.com/ahendriksen/msd_pytorch/compare/v0.8.0...v0.9.0
[0.8.0]: https://www.github.com/ahendriksen/msd_pytorch/compare/v0.7.3...v0.8.0
[0.7.2]: https://www.github.com/ahendriksen/msd_pytorch/compare/v0.7.2...v0.7.3
[0.7.2]: https://www.github.com/ahendriksen/msd_pytorch/compare/v0.7.1...v0.7.2
[0.7.1]: https://www.github.com/ahendriksen/msd_pytorch/compare/v0.6.2...v0.7.1
