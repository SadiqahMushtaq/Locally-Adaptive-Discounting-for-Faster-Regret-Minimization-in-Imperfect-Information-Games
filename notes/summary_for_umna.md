# What we tried (step by step)

1. **Baselines first.** We ran CFR, CFR+ and Discounted CFR (DCFR) on Kuhn and Leduc poker in OpenSpiel. CFR+ beat CFR easily. DCFR lost to CFR+ on Kuhn but beat it on Leduc. So no fixed rule wins everywhere.

2. **Our original idea, LAD-CFR+.** Every information set measures how "jumpy" (volatile) its recent regrets are and discounts old regret more when they are jumpy. We checked that with volatility forced to 0 it gives exactly CFR+, so the code is correct.

3. **First result was weak.** Out of the box, LAD-CFR+ beat CFR+ on Leduc (17%) but lost to DCFR, and lost to CFR+ on Kuhn.

4. **Added Predictive CFR+ (PCFR+) as another baseline.** It was amazing on Kuhn (thousands of times better than CFR+) but the worst method on Leduc.

5. **Tuned LAD-CFR+.** About 40 settings over 3 rounds. The best one (longer memory plus t² averaging) beat DCFR on both Kuhn and Leduc and was best of all on Leduc. But it was tuned on those same two games, so we also tested it on three new games (Liar's Dice with 3 and 4 faces, Goofspiel with 4 cards) without retuning. There it was only okay: good on Liar's Dice-3, worse than plain CFR+ on the other two, and far behind PCFR+.
   - One thing held on all 5 games: the per-information-set version beat the same method with a single game-wide volatility.

6. **New idea, MIX-CFR.** Since every method is good on some games and bad on others, each information set now runs CFR+, DCFR and PCFR+ side by side. A small learner at that information set (regret matching over the three) learns which one works best *there* and plays a weighted mix. There is a simple guarantee: regret at each information set is at most the best method's regret plus a small extra term, so it stays no-regret like CFR.

7. **MIX-CFR results (no tuning at all, all 5 games):**
   - best on Kuhn and Leduc
   - within 1.13× of the best method on Liar's Dice-3, Liar's Dice-4 and Goofspiel. On Liar's Dice-4 the game-wide MIX is best, and on Goofspiel it is essentially tied with PCFR+.
   - better than every method it contains on Leduc and Liar's Dice-4
   - never fails badly, unlike every other method
   - the per-information-set version beat the game-wide version on 3 of 5 games (70× on Kuhn)
   - it costs about 1–1.5× the time per iteration of CFR+

8. **Ablations.**
   - Removing PCFR+ from the mix hurts on all games.
   - Removing CFR+ hurts on 4 of 5.
   - Removing DCFR matters mainly on Leduc.
   - A "predictive" version of the mixing learner did not help.

**Main message for the paper:** adapting the update rule locally, at each information set, works. Adapting only the discount (LAD-CFR+) helps a bit. Letting each information set pick among existing CFR rules (MIX-CFR) is the strongest and most reliable method we tested.

All numbers are in `results/` (`mix/`, `tuning/`, `heldout/`). The convergence figure is `results/figures/convergence.png`.
