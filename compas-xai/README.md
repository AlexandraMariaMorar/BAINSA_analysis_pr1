-Alex's notes
On logistic regression 

The problem it solves
You have a binary outcome — Two_yr_Recidivism is 0 or 1 — and you want to predict the probability of it being 1, given Number_of_Priors, Age_Above_FourtyFive, Age_Below_TwentyFive, Female, Misdemeanor. Plain linear regression is the wrong tool here: it can output numbers like 1.3 or −0.2, which aren't valid probabilities. Logistic regression is built specifically to output something that always lands between 0 and 1.

The linear part
— a weighted sum of your features plus a constant:
z = b₀ + b₁·Number_of_Priors + b₂·Age_Above_FourtyFive + b₃·Age_Below_TwentyFive + b₄·Female + b₅·Misdemeanor
z = the log-odds 
b_i = a coefficient the model learns from data. z can be any real number — still not a probability yet.

The sigmoid: turning z into a probability
z gets squashed through the sigmoid function:
probability = 1 / (1 + e^(−z))
This function has a useful shape: as z → +∞, the output → 1; as z → −∞, the output → 0; at z = 0, the output is exactly 0.5. So no matter what combination of priors and age produces z, the output is guaranteed to be a valid probability. That's the entire trick — everything else is about how z gets built and how the b's get chosen.

Why "log-odds"
Odds and probability are related but different — odds of 3:1 mean probability 0.75. It turns out that if you take the log of the odds, you get back exactly z, the linear sum from above:
log(p / (1 − p)) = z = b₀ + b₁·x₁ + b₂·x₂ + ...
This is why the model is linear on the log-odds scale even though the probability curve is S-shaped.

Interpreting a coefficient
Say b₁ (the coefficient on Number_of_Priors) comes out as 0.15. That means: each additional prior arrest adds 0.15 to the log-odds of recidivism, holding the other features fixed. Log-odds aren't intuitive on their own, so people convert to an odds ratio by exponentiating: e^0.15 ≈ 1.16. That reads as "each additional prior multiplies the odds of recidivism by about 1.16" — a 16% increase in odds per prior.


This is a predictive association, not a causal effect. The model has no idea why priors correlate with the outcome — it's not accounting for confounders, selection effects, or anything else. "More priors → higher odds in this dataset" is not "priors cause recidivism."
If Number_of_Priors isn't standardized (scaled to comparable units with the other 0/1 features), its coefficient is on a different scale than, say, Female's coefficient, and comparing their raw sizes to judge "importance" would be misleading. That's why I standardized it before fitting.
How the b's actually get chosen: maximum likelihood
The model doesn't fit coefficients by minimizing squared error like linear regression does. Instead it picks the b's that make the observed data most probable under the model — formally, maximum likelihood estimation. In practice you don't need to hand-derive this; scikit-learn's optimizer does it. The concept that matters is: the model is graded on whether its predicted probabilities were well-calibrated to what actually happened across the training rows, not on any single prediction.

L1 regularization and sparsity — 
why Plain logistic regression will assign some nonzero coefficient to every feature you give it, even a useless one, just because of noise. L1 regularization adds a penalty to the fitting process for coefficients being large, and it has a distinctive property: it pushes weak coefficients all the way to exactly zero rather than just shrinking them. The practical effect is automatic feature selection — a "sparse" model where only the features that earn their keep survive with a nonzero coefficient. That's the whole reason Person 1's model is called "sparse L1 logistic regression" rather than just "logistic regression."

The strength of that penalty is controlled by C in scikit-learn — counterintuitively, smaller C means stronger regularization (it's an inverse penalty weight), so small C → more coefficients pushed to zero → simpler, sparser model; large C → closer to plain unregularized logistic regression. 



