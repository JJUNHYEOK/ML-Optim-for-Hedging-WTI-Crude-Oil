# Current Task Assignments (2026-07)

Dataset/feature details: [data/README.md](data/README.md). Below is what each
person is doing with that data right now.

## Optimization (동원) — Phase 3
3. Get ready to connect the ML forecast to the hedge decision once it's ready, and pin down the optimization target clearly

## ML (3 people) — same data/split/metrics for everyone, only the model differs !!

**은지 — Random Walk + Ridge (baseline)**
- Compute train/val/test direction accuracy and RMSE for RW (predict "no change") → this is the comparison point for the whole team
- Train Ridge, tune regularization strength on val, evaluate on test only once at the end

**혜원 — Lasso + ElasticNet (feature selection)**
- Check how many of the 26 features actually survive in Lasso → gives the team evidence on which variables are genuinely meaningful
- Compare against ElasticNet to see how much the regularization choice changes performance

**준혁 — LightGBM (nonlinearity / overfitting check)**
- If there's overfitting, try mitigating it with regularization (shallower trees, early stopping, etc.)
- Conclude whether the more complex model actually beats Ridge/Lasso

## Wrap-up
Collect all three results into one table, compare direction accuracy/RMSE against RW, and hand off the best-performing model(s) to the optimization person.
