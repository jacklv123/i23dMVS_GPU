set(FOLDER_NAME "util")

CUDA_ADD_LIBRARY(util
    #bitmap.h bitmap.cc
    camera_specs.h camera_specs.cc
    logging.h logging.cc
    math.h math.cc
    misc.h misc.cc

    
    random.h random.cc
    string.h string.cc
    threading.h threading.cc
    timer.h timer.cc
    version.h version.cc
    FreeImage.h
)


CUDA_ADD_LIBRARY(util_cuda
    cuda.h cuda.cc
    cudacc.h cudacc.cc
)

