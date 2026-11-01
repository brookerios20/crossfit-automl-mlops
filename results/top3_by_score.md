| tool   | feature_set   |   rank | model                                                   | algorithm       |   cv_rmse |   cv_mae |   fit_seconds |   predict_ms_per_row |
|:-------|:--------------|-------:|:--------------------------------------------------------|:----------------|----------:|---------:|--------------:|---------------------:|
| flaml  | all           |      1 | xgboost                                                 | xgboost         |     139.1 |    nan   |       2.926   |            0.002499  |
| flaml  | all           |      2 | xgb_limitdepth                                          | xgb_limitdepth  |     139.3 |    nan   |       0.7112  |            0.0007231 |
| flaml  | all           |      3 | lgbm                                                    | lgbm            |     139.7 |    nan   |       0.3521  |            0.001015  |
| flaml  | top3          |      1 | xgb_limitdepth                                          | xgb_limitdepth  |     167.2 |    nan   |       0.2317  |            0.0002588 |
| flaml  | top3          |      2 | xgboost                                                 | xgboost         |     167.3 |    nan   |       0.4375  |            0.000413  |
| flaml  | top3          |      3 | lgbm                                                    | lgbm            |     167.5 |    nan   |       0.06771 |            0.0001263 |
| h2o    | all           |      1 | StackedEnsemble_AllModels_1_AutoML_1_20261031_182303    | StackedEnsemble |     139.4 |    106.7 |       1.537   |            0.02079   |
| h2o    | all           |      2 | StackedEnsemble_BestOfFamily_1_AutoML_1_20261031_182303 | StackedEnsemble |     139.5 |    106.9 |       1.366   |            0.01752   |
| h2o    | all           |      3 | GBM_5_AutoML_1_20261031_182303                          | GBM             |     140.3 |    107.6 |       0.389   |            0.008459  |
| h2o    | top3          |      1 | StackedEnsemble_AllModels_1_AutoML_2_20261031_182848    | StackedEnsemble |     174.7 |    136.6 |       1.116   |            0.009373  |
| h2o    | top3          |      2 | StackedEnsemble_BestOfFamily_1_AutoML_2_20261031_182848 | StackedEnsemble |     174.8 |    136.7 |       0.966   |            0.006062  |
| h2o    | top3          |      3 | GBM_2_AutoML_2_20261031_182848                          | GBM             |     174.9 |    136.8 |       0.108   |            0.004717  |
