#ifndef DENSITYESTIMATION_H
#define DENSITYESTIMATION_H

#include "trainer.h"

template<class dataT>
class DensityEstimation
{
public:
  typedef int labelT;
  typedef AxisAlignedClassifier<dataT, labelT> ClassifierT;
  typedef GaussianStat<dataT, labelT> GaussianStatisticsT;
  typedef DensityEstimationContext<ClassifierT, dataT, labelT> DensityEstimationContextT;
  typedef DTLeaf<GaussianStatisticsT> LeafT;
  typedef DecisionTree<GaussianStatisticsT, ClassifierT, dataT> DensityTreeT;
  typedef DecisionForest<GaussianStatisticsT, ClassifierT, dataT> DensityForestT;
  typedef Trainer<ClassifierT, GaussianStatisticsT, dataT, labelT> TrainerT;

  typedef MLData<dataT, labelT> TrainingDataT;
  typedef MLData<dataT, GaussianStatisticsT*> TestingDataT;
  typedef Vector<Vector<GaussianStatisticsT*> > TestingResultT;
  typedef Vector<double> DensityT;

  void Learning(TrainingParameters& trainingParameters,
                TrainingDataT& trainingData,
                DensityForestT& forest)
  {
    Random random;
    DensityEstimationContextT densityEstimationTC(trainingData.Dimension());
    TrainerT trainer(trainingData, trainingParameters, densityEstimationTC, random);

    trainer.Training(forest);

    #pragma omp parallel for
    for (index_t i = 0; i < forest.trees_.size(); ++i)
      {
        forest.trees_[i]->partitionFactor_ = 0;
        for (index_t j = 0; j < forest.trees_[i]->nodes_.size(); ++j)
          {
            if (forest.trees_[i]->nodes_[j]->IsLeaf())
              {
                ((LeafT*)forest.trees_[i]->nodes_[j])->statistics_.CalculateSampleNumProportion(trainingData.Size());
                forest.trees_[i]->partitionFactor_ +=
                    ((LeafT*)forest.trees_[i]->nodes_[j])->statistics_.Integration()
                    * ((LeafT*)forest.trees_[i]->nodes_[j])->statistics_.sampleNumProportion_;
//                std::cout << ((LeafT*)forest.trees_[i]->nodes_[j])->statistics_.Integration() << std::endl;
//                std::cout << ((LeafT*)forest.trees_[i]->nodes_[j])->statistics_.sampleNumProportion_ << std::endl;
//                getchar();
              }
          }
      }
  }

  void Predicting(DensityForestT& forest,
                  TestingDataT& testingData,
                  DensityT& probs)
  {
    TestingResultT testingResult;
    forest.Apply(testingData, testingResult);

    size_t treeNum = testingResult.Size();
    size_t dataNum = testingResult[0].Size();

//    for (index_t i = 0; i < dataNum; ++i)
//      {
//        for (index_t j = 0; j < treeNum; ++j)
//          {
//            testingResult[j][i]->Print(0010);
//            std::cout << testingResult[j][i]->sampleNumProportion_ << " " << forest.trees_[j]->partitionFactor_ << std::endl;
//          }
//      }

    probs.Resize(dataNum);

    #pragma omp parallel for
    for (index_t i = 0; i < dataNum; ++i)
      {
        probs[i] = 0;
        for (index_t j = 0; j < treeNum; ++j)
          {
            probs[i] += testingResult[j][i]->Pdf(testingData.data.GetRow(i))
                * testingResult[j][i]->sampleNumProportion_
                / forest.trees_[j]->partitionFactor_;
          }
        probs[i] = probs[i] / treeNum;
      }
  }
};

#endif // DENSITYESTIMATION_H
