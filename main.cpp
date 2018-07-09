#include <iostream>

typedef double dataT;
typedef double labelT;

#include "type.h"
#include "imageio.h"
#include "linearalgebra.h"
#include "integration.h"

void testRegression()
{
  MLData<dataT, labelT>* data = readTextFile<dataT, labelT>("../data/regression/exp2_train.txt", true);
  size_t dataNum = data->Size();
  size_t dataDim = data->Dimension();
  size_t classNum = data->LabelClassNum();

  TrainingParameters params;
  params.treeDepth = 4;
  params.treeNum = 100;
  params.candidateNodeClassifierNum = 10;
  params.candidateClassifierThresholdNum = 10;
  params.subSamplePercent = 0;
  params.splitIG = 0.1;
  params.leafEntropy = 0.05;
  params.verbose = true;
  params.treeType = 2;

  RegressionAxisT regression;
  RegressionForestAxisT forest(params.verbose);
  regression.Learning(params, *data, forest);

  MLData<dataT, labelT>* test = readTextFile<dataT, labelT>("../data/regression/exp2_test.txt", true);
  typedef MLData<dataT, BayesianLinearStatisticsT*> TestingDataT;
  typedef Vector<double> ValueT;

  TestingDataT testingData;
  ValueT results;
  testingData.dataNum_ = test->dataNum_;
  testingData.dataDim_ = test->dataDim_;
  testingData.data = test->data;

  regression.Predicting(forest, testingData, results);

  for (index_t i = 0; i < results.Size(); ++i)
    {
      std::cout << results[i] << std::endl;
    }
}

//void testDensityEstimation()
//{
//  MLData<dataT, labelT>* data = readTextFile<dataT, labelT>("../data/density_estimation/exp1.txt", false);
//  size_t dataNum = data->Size();
//  size_t dataDim = data->Dimension();
//  size_t classNum = data->LabelClassNum();

//  TrainingParameters params;
//  params.treeDepth = 3;
//  params.treeNum = 10;
//  params.candidateNodeClassifierNum = 10;
//  params.candidateClassifierThresholdNum = 10;
//  params.subSamplePercent = 0;
//  params.splitIG = 0.25;
//  params.leafEntropy = 0.05;
//  params.verbose = true;
//  params.treeType = 3;

//  DensityEstimationAxisT densityEstimation;
//  DensityForestAxisT forest(params.verbose);
//  densityEstimation.Learning(params, *data, forest);
////  forest.Print(111);

//  MLData<dataT, labelT>* test = readTextFile<dataT, labelT>("../data/density_estimation/exp1_testData.txt", false);

//  typedef MLData<dataT, GaussianStatisticsT*> TestingDataT;
//  typedef Vector<double> DensityT;

//  TestingDataT testingData;
//  DensityT results;
//  testingData.dataNum_ = test->dataNum_;
//  testingData.dataDim_ = test->dataDim_;
//  testingData.data = test->data;

//  densityEstimation.Predicting(forest, testingData, results);

//  for (index_t i = 0; i < results.Size(); ++i)
//    {
//      std::cout << results[i] << std::endl;
//    }
//}

//void testClassification()
//{
////    MLData<dataT, labelT>* data = readTextFile<dataT, labelT>("../data/classification/exp4_n4.txt", true);
//    MLData<dataT, labelT>* data = readTextFile<dataT, labelT>("../data/classification/R_iris.txt", true);
//    size_t dataNum = data->Size();
//    size_t dataDim = data->Dimension();
//    size_t classNum = data->LabelClassNum();

//  //  std::vector<std::string> filenames;
//  //  filenames.push_back("../data/classification/SimBRATS_HG0001_complete_truth.mha");
//  //  filenames.push_back("../data/classification/SimBRATS_HG0001_T1.mha");
//  //  filenames.push_back("../data/classification/SimBRATS_HG0001_T1C.mha");
//  //  filenames.push_back("../data/classification/SimBRATS_HG0001_T2.mha");
//  //  filenames.push_back("../data/classification/SimBRATS_HG0001_FLAIR.mha");
//  //  typedef itk::Image<dataT, 3> ImageType;
//  //  ImageType::SizeType imageSize;
//  //  size_t dataNum = 0;
//  //  size_t dataDim = 0;
//  //  MLData<dataT, labelT>* data = ImageSeriesReader<ImageType, dataT, labelT>
//  //      (filenames, dataNum, dataDim, imageSize);
//  //  size_t classNum = data->LabelClassNum();
//  //  std::cout << dataNum << " " << dataDim << " " << classNum << std::endl;

//    size_t eachClassSampleNum = 50;
//    MLData<dataT, labelT>* sample = data->Sampling(eachClassSampleNum);

//  //  for (index_t i = 0; i < sample->Size(); ++i)
//  //    {
//  //      for (index_t j = 0; j < dataDim; ++j)
//  //        {
//  //          std::cout << sample->data[i][j] << " ";
//  //        }
//  //      std::cout << sample->label[i] << std::endl;
//  //    }

//    MLData<dataT, HistStatisticsT*> testData(dataNum, dataDim);
//    testData.data = data->data;

//    SoftPredictionT softPrediction(dataNum, classNum);
//    HardPredictionT hardPrediction(dataNum,0);

//  //  double w[] = {0.2, 0.3, 0.1, 0.4, 0.5, 0.6, 0.4};
//    TrainingParameters params;
//    params.treeDepth = 10;
//    params.treeNum = 10;
//    params.candidateNodeClassifierNum = 10;
//    params.candidateClassifierThresholdNum = 10;
//    params.subSamplePercent = 0;
//    params.splitIG = 0.1;
//    params.leafEntropy = 0.05;
//    params.verbose = true;
//  //  for (int i = 0; i < sizeof(w) / sizeof(double); ++i)
//  //    {
//  //      params.weights.push_back(w[i]);
//  //    }

//    ClassificationAxisT classification;
//    ClassificationForestAxisT forest(params.verbose);

//  //  ClassificationLinearT classification;
//  //  ClassificationForestLinearT forest(params.verbose);

//    std::map<index_t, int> mapping;
//    classification.Run(params, *sample, testData, forest, mapping,
//                       softPrediction, hardPrediction);
//  //  forest.Print(1000);
//    size_t misclassifiedNum = 0;
//    for (int i = 0; i < hardPrediction.Size(); ++i)
//      {
//  //      std::cout << hardPrediction[i] << std::endl;
//        if (hardPrediction[i] != data->label[i])
//          {
//            misclassifiedNum++;
//          }
//      }
//    std::cout << "Error rate: " << (double)misclassifiedNum / dataNum << std::endl;
//  //  for (index_t i = 0; i < params.treeNum; ++i)
//  //    {
//  //      std::cout << "tree " << i << " has suspect leaves number = " << forest.trees_[i]->suspectLeaves_ << std::endl;
//  //    }

//  //  std::vector<std::string> outputNames;
//  //  std::stringstream tmpName;
//  //  for (index_t i = 0; i < softPrediction.ColumnSize(); ++i)
//  //    {
//  //      tmpName.str("");
//  //      tmpName << i << ".mha";
//  //      outputNames.push_back(tmpName.str());
//  //      std::cout << outputNames[i] << std::endl;
//  //    }

//  //  ImageSeriesWriter<ImageType>(outputNames, softPrediction, imageSize);
//}

//void testGaussianAggregate()
//{
//  //  MLData<float, int> data(3, 2);
//  //  data.data.PutBack(4);
//  //  data.data.PutBack(2);
//  //  data.data.PutBack(0.5);
//  //  data.data.PutBack(3);
//  //  data.data.PutBack(10);
//  //  data.data.PutBack(8);
//  //  Gaussian<float, int> gaussian(2);
//  //  gaussian.Aggregate(data, 0);
//  ////  gaussian.Aggregate(data, 1);
//  ////  gaussian.Aggregate(data, 2);
//  //  gaussian.CalculateMean();
//  //  gaussian.CalculateCov();
//  //  std::cout << "mean:\n";
//  //  for (int i = 0; i < 2; ++i)
//  //    {
//  //    std::cout << gaussian.mean_[i] << "  ";
//  //    }
//  //  std::cout << "\ncov:\n";
//  //  for (int i = 0; i < 2; ++i)
//  //    {
//  //      for (int j = 0; j < 2; ++j)
//  //        {
//  //          std::cout << gaussian.cov_[i][j] << "  ";
//  //        }
//  //      std::cout << std::endl;
//  //    }
//  //  std::vector<float> d(2);
//  //  d[0] = 1;
//  //  d[1] = 0.5;
//  //  std::cout << "pdf:" << gaussian.Pdf(d) << std::endl;
//}

//void testLinearAlgebra()
//{
//  //  double B[] = {25,  15,  -6,
//  //               15,  18,   0,
//  //               -6,   0,  11  };
//  //  double C[] = {1,2,4};
//  //  double B[] = {18,  22,   54,   42,
//  //               22,  70,   86,   62,
//  //               54,  86,  174,  134,
//  //               42,  62,  134,  106};
//  //  double C[] = {1, 10, 3, 4};
//  //  size_t matSize = sqrt(sizeof(B) / sizeof(double));
//  //  std::vector<double> X(matSize);
//  //  std::vector<double> M(matSize);
//  //  for (index_t i = 0; i < X.size(); ++i)
//  //    {
//  //      X[i] = C[i];
//  //      M[i] = i+8;
//  //    }
//  //  std::vector<std::vector<double> > A;
//  //  A.resize(matSize);
//  //  for (index_t i = 0; i < matSize; ++i)
//  //    {
//  //      A[i].resize(matSize);
//  //      for (index_t j = 0; j < matSize; ++j)
//  //        {
//  //          A[i][j] = B[i*matSize+j];
//  //        }
//  //    }

//  //  std::vector<std::vector<double> > L;
//  //  L.resize(matSize);
//  //  for (index_t i = 0; i < matSize; ++i)
//  //    {
//  //      L[i].resize(matSize);
//  //    }
//  //  std::vector<std::vector<double> > invL;
//  //  invL.resize(matSize);
//  //  for (index_t i = 0; i < matSize; ++i)
//  //    {
//  //      invL[i].resize(matSize);
//  //    }
//  //  MPTimer timer;
//  //  timer.Start();
//  //  cholesky(A, L);
//  //  double result = multiplyXTspdAinvX(X, L, true);
//  //  std::cout << result <<  " spend time: " << timer.StopAndSpendSecond() << std::endl;
//  //  std::cout << "A:\n";
//  //  for (index_t i = 0; i < matSize; ++i)
//  //    {
//  //      for (index_t j = 0; j < matSize; ++j)
//  //        {
//  //          std::cout << std::setw(5) << A[i][j];
//  //        }
//  //      std::cout << std::endl;
//  //    }
//  //  std::cout << "\nL:\n";
//  //  for (index_t i = 0; i < matSize; ++i)
//  //    {
//  //      for (index_t j = 0; j < matSize; ++j)
//  //        {
//  //          std::cout << std::setw(8) << L[i][j];
//  //        }
//  //      std::cout << std::endl;
//  //    }
//  //  std::cout << "\ninvL:\n";
//  //  for (index_t i = 0; i < matSize; ++i)
//  //    {
//  //      for (index_t j = 0; j < matSize; ++j)
//  //        {
//  //          std::cout << std::setw(5) << invL[i][j];
//  //        }
//  //      std::cout << std::endl;
//  //    }
//}

//void testIntegration()
//{
//  //  double a = -1;
//  //  double b = UnivariateNormalCDF(a);
//  //  std::cout.setf(std::ios::fixed);
//  //  std::cout << "UnivariateNormalCDF(" << a << ") = " << std::setprecision(20) << b << std::endl;

//  //  double c = 0.34;
//  //  double d = UnivariateNormalCDFInverse(c);
//  //  std::cout.setf(std::ios::fixed);
//  //  std::cout << "UnivariateNormalCDFInverse(" << c << ") = " << std::setprecision(20) << d << std::endl;

//    int dim = 2;
//    std::vector<double> Mu;
//    Mu.resize(dim);
//    Mu[0] = 0;
//    Mu[1] = -2;
//    std::vector<std::vector<double> > Sigma;
//    Sigma.resize(dim);
//    for (index_t i = 0; i < dim; ++i)
//      {
//        Sigma[i].resize(dim);
//      }
//    for (index_t i = 0; i < dim; ++i)
//      {
//        for (index_t j = 0; j < dim; ++j)
//          {
//            if (i == j)
//              {
//                Sigma[i][j] = 1;
//              }
//            else
//              {
//                Sigma[i][j] = 0;
//              }
//          }
//      }
//    Sigma[0][0] = 2;
//  //  Sigma[0][1] = 1;
//    std::vector<double> a;
//    a.resize(dim);
//    for (index_t i = 0; i < dim; ++i)
//      {
//        a[i] = -INFINITY;
//      }
//  //  a[0] = -1;
//  //  a[1] = 1;
//    std::vector<double> b;
//    b.resize(dim);
//    b[0] = 3;
//    b[1] = 2;
//  //  b[2] = 1;

//    double integral = MultivariateNormalIntegral(Mu, Sigma, a, b, 0.001, 500);
//    std::cout << "MultivariateNormalIntegral = " << std::setprecision(20) << integral << std::endl;
//}




int main()
{
  testRegression();
//  testDensityEstimation();
//  testClassification();
  return 0;
}
