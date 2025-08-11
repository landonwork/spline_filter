# Spline Filter for Conditional Density Estimation

## Summary

This package contains `Spline1D` and `SimpleDensityEstimator` classes for visualizing B-splines and using them
to estimate probability distributions. The most recent commit is a huge refactor moving from a quadratic Bezier
spline filter to a (in my opinion, more functional) B-spline implementation. The current implementation only
supports Tensorflow. Immediate next steps will be creating a Keras layer that will enable a neural network to
output one of these estimated spline distributions and learn in the same way as the `SimpleDensityEstimator`.

This package is the brain child of a data scientist just starting out. Even though there have been no attempts
prove or disprove *why* or *why not*  the spline filter should work, such efforts would be appreciated.
Although it is possible that this invention should not turn out to be practical or correct, the thought is that
feedback from an open-source community can only improve upon what has been put forth.

The spline density estimator was inspired by a combination of factors.

1. The beautiful explanation of splines found [here](https://www.youtube.com/watch?v=jvPPXbo87ds&t=382s&pp=ygUVY29udGludWl0eSBvZiBzcGxpbmVz).
2. The frustrating problem of predicting bids for internet ad space using information available to the seller.
3. The curious lack of understandable conditional density estimators in the field. Either current models
   are trained once and then sampled without being given any inputs other than hypothetical values
   for the stochastic/predicted variables, or I have grossly misunderstood the literature.
4. I needed a final project for [my professor's](https://github.com/sharadkj) machine learning class,
   and I wanted to use a conditional density estimator.

With regards to the license, all rights are reserved by the creator until more research is done.

## Setup

1. Run `pip install -r requirements.txt` from the repo root directory.
2. Open up Jupyter or import what you need! Enjoy!

## Starting Out

Take a look through the `dev.ipynb` notebook to see how this is used.

## License

Copyright © 2025 Landon Work. All rights reserved.
