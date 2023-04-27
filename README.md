# Spline Filter for Conditional Density Estimation

## Summary

This package contains Bayesian(?) bezier spline filter wrappers and functions that can be used in Keras neural networks to create conditional density estimators. The featured inventions are the `SplineFilterLayer` and an accompanying `neg_log_spline_density` loss function. For analyzing the output on a single observation for such a model there is a `Spline` and `SplineFilter` class which allow the user/developer/scientist to visualize and measure (no sampling yet `:(`) the predicted CDF and PDF.

This package is the brain child of a data scientist just starting out. Even though the creator has made no effort as of yet to rigorously work out any proofs of *why* this should work, he would very much like to with an appropriate amount of help and collaboration from his Bayesian brothers. Although it is very possible that this invention should not turn out to be practical or useful, he wanted to expose it to the open-source community on the off-chance that others might take an interest in it and to get experience with publishing a package and and possible to collaborate on something open source.

With regards to the license, the creator is new to open source package development and publication, so all rights are reserved until he learns more about it.

## Installation

For now, the easiest way to use the package is to install Poetry 1.2+ and run `poetry install` in the root directory.
The creator plans to publish the package to PyPa.

## Starting Out

TODO

## Issues

The creator will do his best to list known issues and hypothetical features here and keep it updated.

- TODO

## License

Copyright © 2023 Landon Work. All rights reserved.
