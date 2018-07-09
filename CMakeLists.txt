cmake_minimum_required (VERSION 2.6)
project (randomForest)

# Setting the specific C compiler in Shell environment, e.g. export CC=gcc
# Instead of set up here: http://www.cmake.org/Wiki/CMake_FAQ#How_do_I_use_a_different_compiler.3F
#set(CMAKE_C_COMPILER gcc)
#set(CMAKE_CXX_COMPILER g++)

#find_package(MPI)
#if(MPI_FOUND)
#    add_definitions(-DMPI_INSTALLED)
#    set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
#    set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
#    include_directories(${MPI_INCLUDE_PATH})
#    set(LIBRARIES ${LIBRARIES} ${MPI_LIBRARIES})
#endif(MPI_FOUND)

find_package(OpenMP)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)

find_package(ITK)
if(ITK_FOUND)
    include(${ITK_USE_FILE})
    set(LIBRARIES ${LIBRARIES} ${ITK_LIBRARIES})
endif(ITK_FOUND)

set (SOURCES
    main.cpp
)

set (HEADERS
    node.h
    data.h
    tree.h
    forest.h
    random.h
    trainer.h
    classifier.h
    statistics.h
    trainingcontext.h
    type.h
    utility.h
    classification.h
    regression.h
    densityestimation.h
    linearalgebra.h
    integration.h
)

if(ITK_FOUND)
set (HEADERS
    ${HEADERS}
    imageio.h
)
endif(ITK_FOUND)

add_executable (randomForest ${SOURCES} ${HEADERS})
target_link_libraries(randomForest ${LIBRARIES})
