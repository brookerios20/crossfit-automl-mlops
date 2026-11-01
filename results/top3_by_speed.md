| tool   | feature_set   |   rank | model                                       | algorithm   |   cv_rmse |   cv_mae |   fit_seconds |   predict_ms_per_row |
|:-------|:--------------|-------:|:--------------------------------------------|:------------|----------:|---------:|--------------:|---------------------:|
| flaml  | all           |      1 | rf                                          | rf          |     149.2 |    nan   |       0.04415 |            0.0006939 |
| flaml  | all           |      2 | extra_tree                                  | extra_tree  |     145.3 |    nan   |       0.2012  |            0.0006632 |
| flaml  | all           |      3 | sgd                                         | sgd         |     162.9 |    nan   |       0.3032  |            0.0002144 |
| flaml  | top3          |      1 | extra_tree                                  | extra_tree  |     168.9 |    nan   |       0.0174  |            0.0006921 |
| flaml  | top3          |      2 | lgbm                                        | lgbm        |     167.5 |    nan   |       0.06771 |            0.0001263 |
| flaml  | top3          |      3 | rf                                          | rf          |     167.5 |    nan   |       0.1     |            0.001291  |
| h2o    | all           |      1 | GLM_1_AutoML_1_20261031_182303              | GLM         |     148.4 |    113.7 |       0.143   |            0.00123   |
| h2o    | all           |      2 | GBM_grid_1_AutoML_1_20261031_182303_model_1 | GBM         |     141.2 |    108.2 |       0.337   |            0.007402  |
| h2o    | all           |      3 | GBM_2_AutoML_1_20261031_182303              | GBM         |     140.5 |    107.7 |       0.372   |            0.008046  |
| h2o    | top3          |      1 | GLM_1_AutoML_2_20261031_182848              | GLM         |     183.3 |    143.6 |       0.045   |            0.000792  |
| h2o    | top3          |      2 | GBM_5_AutoML_2_20261031_182848              | GBM         |     175   |    136.8 |       0.101   |            0.004483  |
| h2o    | top3          |      3 | GBM_grid_1_AutoML_2_20261031_182848_model_1 | GBM         |     175.1 |    137   |       0.107   |            0.004782  |
