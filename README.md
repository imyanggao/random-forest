# random-forest
  This code is a generic random forest library and could be used for classification, regression and density estimation tasks. The library uses OpenMP for multithreading to speed up the training and inference. Back to 2014, the initial purpose of this code is to develop an image segmentation algorithm with multi-modality support for itk-SNAP project (http://www.itksnap.org). The code is inspired by the book:
  * A. Criminisi, J. Shotton, and E. Konukoglu, Decision Forests: 
   A Unified Framework for Classification, Regression, Density Estimation, 
   Manifold Learning and Semi-Supervised Learning. Foundations and Trends in 
   Computer Graphics and Computer Vision. NOW Publishers. Vol.7: No 2-3, pp 81-227. 2012.

## Prerequisites
  All codes are written in C++. During developing, the author tried to avoid any library dependence (which means any numerical linear algebra algorithms in use are written from scratch). The CMake is used for managing the build process of software. If want to compile without any modification, please also add two optional libraries list below:
  * OpenMP (optional) for multithreading
  * ITK (optional) for medical image IO