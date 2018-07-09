#ifndef TYPE_H
#define TYPE_H

#include "classification.h"
#include "densityestimation.h"
#include "regression.h"

typedef AxisAlignedClassifier<dataT, labelT> AxisClassifierT;
typedef LinearClassifier<dataT, labelT> LinearClassifierT;
typedef HistogramStat<dataT, labelT> HistStatisticsT;
typedef GaussianStat<dataT, labelT> GaussianStatisticsT;
typedef BayesianLinearStat<dataT, labelT> BayesianLinearStatisticsT;
typedef DecisionForest<HistStatisticsT, AxisClassifierT, dataT> ClassificationForestAxisT;
typedef DecisionForest<HistStatisticsT, LinearClassifierT, dataT> ClassificationForestLinearT;
typedef DecisionForest<GaussianStatisticsT, AxisClassifierT, dataT> DensityForestAxisT;
typedef DecisionForest<BayesianLinearStatisticsT, AxisClassifierT, dataT> RegressionForestAxisT;
typedef Classification<dataT, AxisClassifierT> ClassificationAxisT;
typedef Classification<dataT, LinearClassifierT> ClassificationLinearT;
typedef DensityEstimation<dataT> DensityEstimationAxisT;
typedef Regression<dataT, AxisClassifierT> RegressionAxisT;
typedef Matrix<double> SoftPredictionT;
typedef Vector<labelT> HardPredictionT;

#endif // TYPE_H
