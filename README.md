# CSP-Bayes

## Intro
This Python project explores the connections between palate preferences, enrollment, and the concept of being awesome. By modeling these relationships as a Bayesian network, the program aims to estimate various probabilities based on different conditions.

## How it Works
The project defines a Bayesian network with nodes representing Palate (P), Enthusiasm (E), Moxie (M), and Awesomeness (A). Each node's state is influenced by its parent nodes, and their conditional probabilities are specified in the code.

The project provides three sampling methods to estimate probabilities:

Simple Sampler: Generates samples without any evidence.
Rejection Sampler: Generates samples given evidence using rejection sampling technique.
Likelihood Weighting Sampler: Generates samples given evidence using likelihood weighting method.
Usage
The code can be used to estimate probabilities for different scenarios, such as whether someone is enrolled (E=True), given other conditions. The project also includes functions to compare and visualize the convergence of different sampling methods as the number of samples increases.

## How to Run
Requirements: Ensure you have Python installed on your system.
Download: Download the project files.
Run the Code: Execute the Python script to see the probabilities estimated by different sampling methods.
Examples
Enrollment Probability: Estimate the probability of being enrolled.


compare_estimates({'E': True}, {}, num_samples, sampler_simp, sampler_reject, sampler_like)
Moxie Probability Given -Palate: Estimate the probability of having Moxie given the absence of palate preference.

compare_estimates({'M': True}, {'P': False}, num_samples, sampler_simp, sampler_reject, sampler_like)
Awesomeness Probability: Estimate the probability of being awesome.

compare_estimates({'A': True}, {}, num_samples, sampler_simp, sampler_reject, sampler_like)
Plotting
The project includes a function to create a plot comparing the convergence of rejection sampling and likelihood weighting methods as the number of samples increases. The plot is saved as bayes_awesome.pdf.

## Note
This project is an educational exploration of Bayesian networks and probability estimation methods. The results provided by the sampling methods are empirical estimates and may not represent real-world scenarios accurately.
