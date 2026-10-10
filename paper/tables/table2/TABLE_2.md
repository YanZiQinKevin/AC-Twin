# Table 2 | Information roles and permitted access.

| Information role | Permitted use | Prohibited use |
| --- | --- | --- |
| Inference-legal signals | Predictions, base selection, features and risk ranking for the evaluated group | Any true counterfactual consequence or truth-derived quantity for that group |
| Training-only supervision | Fit risk targets and training state within permitted development-training roles | Confirmation labels for fitting, configuration selection or normalization |
| Frozen model outputs | Fit on permitted development train, select/calibrate on allowed monitor/validation, then freeze; outputs may enter evaluated-group inference | Train, select or update upstream models on confirmation data, or embed evaluated-group truth in predictions |
| Evaluation-only truth | Regret, response strata, qualification and intervals after scoring is frozen | Feed retrospective evaluation information back into evaluated-group inference |


Source: the corresponding manuscript table. Definitions and numerical values are unchanged; numbering was unified.
