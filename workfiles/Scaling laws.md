# Standard Deviation of the Sample Mean

## Setup

Let

$$
X_1, X_2, \dots, X_m
$$

be independent and identically distributed (i.i.d.) random variables with:

- Mean: $\mu$  
- Variance: $\sigma^2$  
- Standard deviation: $\sigma$  

Define the **sample mean**:

$$
\bar X = \frac{1}{m}\sum_{i=1}^m X_i
$$

Important:  
$\bar X$ is itself a **random variable**, because every new dataset produces a new mean.

Our goal is to compute:

$\text{std}(\bar X)$

---

## Population Variance vs Samples

Variance is defined over a **distribution**, not a single observed value:

$$
\mathrm{Var}(X)=\mathbb E[(X-\mu)^2]=\sigma^2
$$

Each $X_i$ is a draw from the same distribution, therefore:

$$
\mathrm{Var}(X_i)=\sigma^2
$$

After observing data $x_i$, those are just numbers.  
They have no variance anymore — only the underlying random variables do.

---

## Derivation

Start with variance of the mean:

$$
\mathrm{Var}(\bar X)
=
\mathrm{Var}\left(\frac{1}{m}\sum_{i=1}^m X_i\right)
$$

Pull out the constant:

$$
=
\frac{1}{m^2}\mathrm{Var}\left(\sum_{i=1}^m X_i\right)
$$

Because the variables are independent:

$$
\mathrm{Var}\left(\sum_{i=1}^m X_i\right)
=
\sum_{i=1}^m \mathrm{Var}(X_i)
=
m\sigma^2
$$

So:

$$
\mathrm{Var}(\bar X)=\frac{m\sigma^2}{m^2}=\frac{\sigma^2}{m}
$$

Take square root:

$$
\text{std}(\bar X)=\frac{\sigma}{\sqrt m}
$$

---

## Interpretation

There exists a **distribution of sample means** (sampling distribution).

Each experiment:

1. Draw $m$ samples  
2. Compute mean  
3. Record it  

Repeating forever produces a histogram centered at $\mu$.

Its width is:

$$
\frac{\sigma}{\sqrt m}
$$

This measures how much the estimated mean typically deviates from the true mean.

---

## Key Result

$$
\boxed{\text{std(mean)}=\frac{\sigma}{\sqrt m}}
$$

---

## Intuition

- Individual samples have noise $\sigma$  
- Averaging cancels noise  
- Variance shrinks as $1/m$  
- Standard deviation shrinks as $1/\sqrt m$  

So:

- 4× data → half the error  
- 100× data → 10× less error  

Progress is slow: improvements are sublinear.

---

## Important Nuance

More data does **not** guarantee every new estimate is closer.

It guarantees that **expected error decreases**.

This holds if:

- samples are unbiased  
- variance is finite  
- samples are independent (or weakly correlated)  

---

## Big Picture

The sample mean is an estimator.

Its uncertainty obeys a universal square-root law:

$$
\text{uncertainty} \propto \frac{1}{\sqrt m}
$$

This principle appears in:

- statistics  
- Monte Carlo methods  
- stochastic gradient descent  
- learning curves  


