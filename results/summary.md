# Results summary

CV metrics use the same 5-fold split of the training set. Test metrics use the held-out 20% test split. The top-3 runs use the features ranked highest by the same tool's all-features run.

## Top 5 features (all-features runs)

| tool   |   rank | feature                     |   rmse_increase |   rmse_increase_std |   importance_share |
|:-------|-------:|:----------------------------|----------------:|--------------------:|-------------------:|
| flaml  |      1 | gender                      |         111.7   |              2.429  |          nan       |
| flaml  |      2 | bmi                         |          21.27  |              0.4754 |          nan       |
| flaml  |      3 | age                         |          20.05  |              0.7241 |          nan       |
| flaml  |      4 | weight                      |          16.23  |              1.082  |          nan       |
| flaml  |      5 | howlong__less_than_6_months |           3.574 |              0.2542 |          nan       |
| h2o    |      1 | gender                      |         nan     |            nan      |            0.4613  |
| h2o    |      2 | weight                      |         nan     |            nan      |            0.2248  |
| h2o    |      3 | bmi                         |         nan     |            nan      |            0.07433 |
| h2o    |      4 | age                         |         nan     |            nan      |            0.04125 |
| h2o    |      5 | height                      |         nan     |            nan      |            0.03805 |

## Top 3 models by CV RMSE

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

## Top 3 models by fit time

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

## Best models against the Assignment 2 model

| run          | model                                                | algorithm                     |   cv_rmse |   test_rmse |   test_mae |   test_r2 |   fit_seconds |   predict_ms_per_row |   search_seconds |
|:-------------|:-----------------------------------------------------|:------------------------------|----------:|------------:|-----------:|----------:|--------------:|---------------------:|-----------------:|
| baseline_all | assignment2_v2_hp_a                                  | HistGradientBoostingRegressor |     140.7 |       141.3 |      108.1 |    0.7414 |        0.139  |            0.0009045 |            nan   |
| flaml_all    | xgboost                                              | xgboost                       |     139.1 |       140.2 |      106.9 |    0.7457 |        2.926  |            0.002499  |            302.4 |
| flaml_top3   | xgb_limitdepth                                       | xgb_limitdepth                |     167.2 |       167.2 |      129.8 |    0.6379 |        0.2317 |            0.0002588 |            300.2 |
| h2o_all      | StackedEnsemble_AllModels_1_AutoML_1_20261031_182303 | StackedEnsemble               |     139.4 |       140   |      107.1 |    0.7463 |        1.537  |            0.02079   |            327.4 |
| h2o_top3     | StackedEnsemble_AllModels_1_AutoML_2_20261031_182848 | StackedEnsemble               |     174.7 |       176.1 |      137.9 |    0.5984 |        1.116  |            0.009373  |            245.1 |
