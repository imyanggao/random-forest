#ifndef UTILITY_H
#define UTILITY_H

#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <time.h>
#include "omp.h"
#include "data.h"

template<class T>
void readBasicType(std::istream& is, T& obj)
{
  is.read((char*)(&obj), sizeof(T));
}

template<class T>
void writeBasicType(std::ostream& os, const T& obj)
{
  os.write((const char*)(&obj), sizeof(T));
}

// paul: bug, this needs to be inline
inline int getLineNum(const std::string& name)
{
  std::ifstream inFile(name.c_str(), std::ios::in);
  return std::count(std::istreambuf_iterator<char>(inFile),
                    std::istreambuf_iterator<char>(), '\n');
  inFile.close();
}

template<class dataT, class labelT>
MLData<dataT, labelT>* readTextFile(const std::string& name, bool labeled = false)
{
  int dataNum = getLineNum(name);
  int dataDim = 0;
  bool first = true;
  MLData<dataT, labelT>* data;
  labelT tmpLabel = 0;
  dataT tmpData = 0;
  size_t cLineIdx = 0;
  index_t ctokenIdx = 0;
  std::string line;
  std::ifstream inFile(name.c_str(), std::ios::in);
  if (inFile.is_open())
    {
      while ( getline(inFile, line) )
        {
          std::istringstream iss(line);
          std::vector<std::string> tokens;
          std::copy(std::istream_iterator<std::string>(iss),
                    std::istream_iterator<std::string>(),
                    std::back_inserter<std::vector<std::string> >(tokens));
          if (first)
            {
              if (labeled)
                {
                  dataDim = tokens.size() - 1;
                }
              else
                {
                  dataDim = tokens.size();
                }
              data = new MLData<dataT, labelT>(dataNum, dataDim);
              first = false;
            }

          ctokenIdx = 0;

          if (labeled)
            {
              if ((tokens.size() - 1) == dataDim)
                {
                  std::stringstream(tokens[ctokenIdx++]) >> tmpLabel;
                  data->label[cLineIdx] = tmpLabel;
                }
              else
                {
                  data->label[cLineIdx] = -HUGE_VAL;
                }
//              std::cout << data->label[cLineIdx] << std::endl;
            }

          for (index_t i = 0; ctokenIdx < tokens.size(); ++ctokenIdx, ++i)
            {
              std::stringstream(tokens[ctokenIdx]) >> tmpData;
              data->data[cLineIdx][i] = tmpData;
//              std::cout << data->data[cLineIdx][i] << " ";
            }
//          std::cout << std::endl;
          cLineIdx++;
        }
      inFile.close();
    }
  else
    {
      throw std::runtime_error("open file error");
    }
  return data;
}

class Timer
{
public:
  void Reset(void)
  {
    start_ = 0;
    finish_ = 0;
  }

  void Start(void)
  {
    start_ = clock();
  }
  void Stop(void)
  {
    finish_ = clock();
  }
  double SpendSecond(void)
  {
    return (double)(finish_-start_)/CLOCKS_PER_SEC;
  }
  double StopAndSpendSecond(void)
  {
    Stop();
    return SpendSecond();
  }

private:
  clock_t start_, finish_;
};

class MPTimer
{
public:
  void Reset(void)
  {
    start_ = 0.0;
    finish_ = 0.0;
  }

  void Start(void)
  {
    start_ = omp_get_wtime();
  }
  void Stop(void)
  {
    finish_ = omp_get_wtime();
  }
  double SpendSecond(void)
  {
    return finish_-start_;
  }
  double StopAndSpendSecond(void)
  {
    Stop();
    return SpendSecond();
  }

private:
  double start_, finish_;
};

#endif // UTILITY_H
