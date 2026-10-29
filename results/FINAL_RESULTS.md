# Final competition results

**Competition:** [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
**Metric:** macro-average ROC AUC across twelve findings  
**Competition close:** 22 October 2026  
**Result source:** Kaggle submissions endpoint and final selection screenshot, reviewed 27 October 2026.

## Four archived entries

| Submission ref | Date (UTC) | Entry | Public AUC | Private AUC | Final selection |
| ---: | --- | --- | ---: | ---: | :---: |
| `57292701` | 2026-10-17 | R6 Mirror R40 — compact-forward fixed 40/60 rank blend | **0.958** | **0.942** | No |
| `57433412` | 2026-10-20 | R6 A — R6 plus ConvNeXt MRI, no medical correction | **0.958** | **0.942** | No |
| `57485947` | 2026-10-21 | Final R6 B — shared trained weights and fixed medical correction | **0.958** | **0.942** | **Yes** |
| `57531062` | 2026-10-22 | R6 Raptor blend without anatomy correction | **0.958** | **0.942** | **Yes** |

The top team result among these was a tie at **0.942 private AUC**; public AUC was **0.958**. The final selection was the R6 B medical-correction variant and the Raptor variant without anatomy correction. This archive does not infer a final private rank from scores alone.

## Additional tied result

The authenticated submission listing also showed the separate **R6 Raptor Anatomy-Knowledge Correction** entry (`57530038`) at **0.958 public / 0.942 private**. It is not one of the four notebook snapshots requested for this repository pass and was not one of the two rows marked selected in the supplied screenshot.

## Earlier best

Before the final R6 submissions, the team's E030 Compact35 entry (`57090331`) scored **0.954 public / 0.938 private**. The final R6 entries improved on that public and private result by 0.004 each. Internal Gold-58 and report-holdout numbers in the research archive are development diagnostics and must not be conflated with leaderboard AUC.

The public leaderboard's last accessible snapshot displayed a highest score of **0.960**. The snapshot does not establish our final private position. Scores are competition metrics, not clinical validation.
