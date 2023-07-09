# Spline Filter for Conditional Density Estimation

## Setup

To exactly reconstruct the environment used in development, please follow these steps.

If you haven't installed poetry yet, run this in your terminal:
- `pip install poetry`

Then run the following commands:
1. poetry install
2. poetry shell
3. python ipykernel --user --name kernel_name

This will create the Poetry virtual environment with all the package dependencies and install it as a Jupyter
kernel that the notebooks can use.

Users should not be shy! Anyone can create an issue or a pull request!

## Summary

This package contains Bayesian(?) bezier spline filter wrappers and functions that can be used in Keras
neural networks to create conditional density estimators which take any number of inputs and return the
conditional CDF of a single variable. The featured inventions are the `SplineFilterLayer`
and an accompanying `neg_log_spline_density` loss function. For analyzing the output on a single observation
for such a model there is a `Spline`, `OneToOneSpline`, and `SplineFilter` class which allow the
user/developer/scientist to visualize and measure (no sampling yet `:(`) the predicted CDF and PDF.

This package is the brain child of a data scientist just starting out. Even though there have been no attempts
prove or disprove *why* or *why not*  the spline filter should work, such efforts would be appreciated.
Although it is possible that this invention should not turn out to be practical or correct, the thought is that
feedback from an open-source community can only improve upon what has been put forth.

The spline filter was inspired by a combination of factors.

1. The beautiful explanation of splines found [here](https://www.youtube.com/watch?v=jvPPXbo87ds&t=382s&pp=ygUVY29udGludWl0eSBvZiBzcGxpbmVz).
2. The frustrating problem of predicting bids for internet ad space using information available to the seller.
3. The curious lack of understandable conditional density estimators in the field. Either current models
   are trained once and then sampled without being given any inputs other than hypothetical variables
   for the stochastic/predicted variables, or I have grossly misunderstood the literature.
4. I needed a final project for [my professor's](https://github.com/sharadkj) machine learning class,
   and I wanted to use a conditional density estimator.

With regards to the license, all rights are reserved by the creator until more research is done.

## Starting Out

TODO

## Issues

The creator will do his best to list known issues and hypothetical features here and keep it updated.

- TODO

## License

Copyright © 2023 Landon Work. All rights reserved.
