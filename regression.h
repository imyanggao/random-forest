#ifndef REGRESSION_H
#define REGRESSION_H

#include "trainer.h"

template<class dataT, class ClassifierT>
class Regression{
public:
  typedef double labelT;
  typedef BayesianLinearStat<dataT, labelT> BayesianLinearStatisticsT;
  typedef RegressionContext<ClassifierT, dataT, labelT> RegressionContextT;
  typedef DecisionForest<BayesianLinearStatisticsT, ClassifierT, dataT> DecisionForestT;
  typedef Trainer<ClassifierT, BayesianLinearStatisticsT, dataT, labelT> TrainerT;

  typedef MLData<dataT, labelT> TrainingDataT;
  typedef MLData<dataT, BayesianLinearStatisticsT*> TestingDataT;
  typedef Vector<Vector<BayesianLinearStatisticsT*> > TestingResultT;
  typedef Vector<labelT> ValueT;

  void Learning(TrainingParameters& trainingParameters,
                TrainingDataT& trainingData,
                DecisionForestT& forest)
  {
    Random random;
    RegressionContextT regressionTC(trainingData.Dimension());
    TrainerT trainer(trainingData, trainingParameters, regressionTC, random);

    trainer.Training(forest);
  }

  void Predicting(DecisionForestT& forest,
                  TestingDataT& testingData,
                  ValueT& values)
  {
    TestingResultT testingResult;
    forest.Apply(testingData, testingResult);

    size_t treeNum = testingResult.Size();
    size_t dataNum = testingResult[0].Size();
    values.Resize(dataNum);
    std::vector<labelT> tmpMean;
    std::vector<labelT> tmpVar;
    tmpMean.resize(dataNum);
    tmpVar.resize(dataNum);

    // before parallel computing, must pre-calculate A inverse to avoid race condition
    for (index_t i = 0; i < dataNum; ++i)
      {
        for (index_t j = 0; j < treeNum; ++j)
          {
            testingResult[j][i]->CalculateAInverse();
          }
      }

    #pragma omp parallel for
    for (index_t i = 0; i < dataNum; ++i)
      {
        values[i] = 0;
        for (index_t j = 0; j < treeNum; ++j)
          {
            testingResult[j][i]->Predict(
                  testingData.data.GetRow(i), tmpMean[i], tmpVar[i]);
            values[i] += tmpMean[i];
          }
        values[i] = values[i] / treeNum;
      }

  }
};


#endif // REGRESSION_H
