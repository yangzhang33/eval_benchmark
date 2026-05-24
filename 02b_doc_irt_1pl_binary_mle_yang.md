# 1PL IRT Modeling with MLE

Traditionally, Classical Test Theory (CTT) evaluates the performance of examinee (LLMs in our case) by its overall accuracy on a set of problems (MCQs in our case). However this leads to a chicken-and-egg situation: the difficulty of problems affects the evaluation of examinees' ability, while the examinees' ability affects our judgement on problem difficulty.

## 1PL IRT (Rasch Model)

The 1PL IRT (One-Parameter Logistic Item Response Theory) was proposed to address this issue. It models $P_{ij}$, the probability of an examinee $j$ to answer correctly one problem $i$, by both the examinee's ability $\theta_j$ and the problem difficulty $b_i$:

$$P_{i}(\theta_j)=\frac{1}{1+\exp(-(\theta_j-b_i))}\coloneqq\sigma(\theta_j-b_i).$$

Under this modeling, the difficulty $b_i$ is the only factor affecting the shape of the Item Characteristic Curve (ICC) $P_i: \theta_j\mapsto[0,1]$.

In our experimental settings, every logic problem $i$ is considered to have a universal difficulty $b_i$ no matter the facet (translated language) to isolate the influence of the prompt from the underlying difficulty of the problem itself. Meanwhile, every LLM $j$ is considered to have one ability parameter $\theta_{j, \text{CA/CS}, \text{language}}$ per subset per prompt language. The design aims to highlight the ability gap of LLMs between culture-specific and culture-agnostic subsets, as well as the influence of the prompt language on this gap.

## Maximum Likelihood Estimation

In contrast to the No-U-Turn Sampler (NUTS), which explores the full posterior parameter distribution, Maximum Likelihood Estimation (MLE) converges the model to a single point estimate that maximizes the likelihood of the observed responses. We treat the LLM ability factors $\theta$ and the problem difficulties $b$ as free, learnable parameters with no prior imposed on either, and optimize them by Stochastic Gradient Descent (SGD) — concretely, the Adam optimizer. The trade-off is deliberate: MLE is fast and yields a canonical point estimate, but by design it loses the track of posterior uncertainty and confidence estimation that NUTS provides directly.

The likelihood of a single observation $y_{ij}$, where $y=1$ represents a correct answer and $y=0$ an incorrect one, is identical to the NUTS formulation:

$$P(y_{ij}|\theta_j,b_i)=\sigma(\theta_j-b_i)^{y_{ij}}(1-\sigma(\theta_j-b_i))^{(1-y_{ij})}.$$

The log-likelihood on the entire benchmark observation is thus:

$$\log P(Y|\theta,b)=\sum_{ij}\Big[y_{ij}\log\sigma(\theta_j-b_i)+(1-y_{ij})\log(1-\sigma(\theta_j-b_i))\Big],$$

$$=\sum_{ij}\Big[y_{ij}(\theta_j-b_i)-\log(1+\exp(\theta_j-b_i))\Big].$$

The MLE objective is to maximize this quantity, equivalently to minimize the negative log-likelihood (NLL), which is exactly the summed binary cross-entropy on the logits $\theta_j-b_i$:

$$(\hat\theta, \hat b)=\arg\max_{\theta, b}\log P(Y|\theta, b)=\arg\min_{\theta, b}\big[-\log P(Y|\theta, b)\big].$$

Because no prior is imposed, the gradient of the objective contains only the data-fit term and, unlike the NUTS log-posterior, carries no quadratic penalty on $\theta$ or $b$:

$$\frac{\partial\log P}{\partial\theta_j}=\sum_i(y_{ij}-P_{ij}),\quad \frac{\partial\log P}{\partial b_i}=\sum_j(P_{ij}-y_{ij}).$$

Adam follows these gradients with an adaptive, momentum-distorted step. In our approach we run a single optimization trajectory at learning rate $0.05$ for up to $5000$ steps, terminating early once the relative change of the loss over a sliding window of $100$ steps falls below $10^{-5}$. The result is one converged set of point estimates $(\hat\theta, \hat b)$, not a distribution.

## Identifiability and Per-Step Anchoring

The 1PL likelihood is invariant under a joint location shift: replacing $\theta_j\to\theta_j+c$ and $b_i\to b_i+c$ leaves every logit $\theta_j-b_i$ — and therefore the likelihood — unchanged. (There is no scale degeneracy here, since the 1PL model fixes the discrimination to unity.) Where NUTS resolves this translation invariance softly through the unit-Gaussian prior $\theta_j\sim\mathcal{N}(0,1)$, the unregularized MLE has nothing to pin down the location, so the optimizer would otherwise drift freely along this null direction.

We therefore eliminate the unidentified degree of freedom by a hard anchor applied after every Adam step. The shift is applied per *connected component* of the item–subset graph: the single culture-agnostic (CA) component, whose `ca::` items appear in every `*_ca` subset, and one culture-specific (CS) component per language, whose `cs::{lang}::` items are shared only between `{lang}_cs` and `{lang}_cs_en`. Two subsets in different components share no item and so are completely separable. Within each component $C$ we compute

$$\text{shift}_C=\operatorname{mean}_{(j, s)\in C}\theta_{j, s},\qquad \theta_{j, s}\leftarrow\theta_{j, s}-\text{shift}_C,\quad b_i\leftarrow b_i-\text{shift}_C\;\;(i\in C).$$

Subtracting the same constant from $\theta$ and $b$ preserves $\theta_j-b_i$ exactly, so the likelihood is untouched, while the convention $\operatorname{mean}(\theta_C)=0$ is enforced at every step. The converged estimate is thus canonical and directly comparable across components.

## From Point Estimate to Uncertainty

The fit produces a single $(\hat\theta, \hat b)$, persisted as a NetCDF with a degenerate `(chain=1, draw=1)` shape so that downstream tooling shared with the NUTS pipeline can read it uniformly. All distributional summary columns (standard deviation, percentiles, HDI, margin of error, R-hat, ESS) collapse under this shape, so the analysis stage emits only the `*_mean` point estimates for $\theta$ and $b$.

This is the central difference from NUTS. The Bayesian sampler reports a $95\%$ credible interval $[q_{2.5\%}, q_{97.5\%}]$ for every quantity directly from posterior draws; the MLE point estimate, taken alone, carries no such uncertainty. Recovering a frequentist confidence interval requires a separate step — propagating the curvature of the log-likelihood at $(\hat\theta, \hat b)$ through the Fisher information and its Schur complement onto the $\theta$ contrasts of interest. That construction is documented for the binary 2PL MLE in `06_mle_schur_fisher_ci.md`; the 1PL binary MLE pipeline here stops at the point estimate.
