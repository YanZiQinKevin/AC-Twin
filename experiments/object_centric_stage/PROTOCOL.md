# Object-centric stage protocol

## Purpose

Replace distance-sorted observation rows with stable episode-local vehicle identities before introducing any graph architecture.

## Identity definition

- Entity 0 is the controlled ego vehicle.
- Remaining entities follow the simulator's deterministic, seeded creation order at reset.
- The identity is episode-local; no claim is made that IDs correspond across unrelated episodes.
- Facts and counterfactual branches must have identical initial fingerprints: entity ID, vehicle class, initial pose, speed and planned route.
- Newly spawned vehicles without a registered identity cause an explicit failure rather than silent row reassignment.

## Object state

Each entity row contains: presence, ego flag, absolute x/y, velocity x/y, and heading sine/cosine. Missing entities use a zero row and ID -1. Model-side episode anchoring and normalization will be fitted on the training split only.

## Audit gate

For both `merge-v0` and `roundabout-v0`, across 100 seeded episode pairs:

1. No initial fingerprint mismatch.
2. No unregistered vehicle during rollout.
3. Zero entity-ID mismatch across identical prefixes.
4. Maximum state difference across identical prefixes no greater than 1e-6.
5. Record how often distance-sorted row ranks change, demonstrating why the previous row representation is unsuitable for node-level causal propagation.

Graph-model implementation begins only after this gate passes.
