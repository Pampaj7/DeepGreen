# Pseudo-replication caveat

IMPORTANT -- what these tests can and cannot establish.

Each of the 48 configurations was executed exactly ONCE. The 30 epochs of a run
are repeated measurements of the same run, not independent replications: they
share one process, one weight initialisation, one allocator state, one JIT
outcome and one thermal trajectory. The effective number of independent
observations per configuration is therefore 1, not 30.

The tests below are computed over epochs and are reported for completeness and
comparability with the prior literature, but their p-values are anti-conservative
by an unknown factor: they test whether the epochs of run A differ from the
epochs of run B, not whether ecosystem A differs from ecosystem B. Effect sizes
(Cliff's delta) and the observed dispersion are the more informative outputs.

Any claim of a statistically significant *ecosystem* difference requires
independent run-level repetitions. results/analysis/repetition_protocol.md
specifies the protocol; scripts/run_campaign.py implements it.
