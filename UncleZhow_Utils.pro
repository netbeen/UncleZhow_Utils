QT += core xml
QT -= gui

TARGET = UncleZhow_Utils
CONFIG += console
CONFIG += c++11


TEMPLATE = app

SOURCES += main.cpp \
    watershedprocessor.cpp \
    thresholdprocessor.cpp \
    nnhdprocessor.cpp \
    findsimilarpoint.cpp \
    patchmatchprocessor.cpp \
    mygabor.cpp \
    utils.cpp \
    grabcut.cpp \
    graphcut.cpp \
    gco-v3.0/GCoptimization.cpp \
    gco-v3.0/graph.cpp \
    gco-v3.0/LinkedBlockList.cpp \
    gco-v3.0/maxflow.cpp \
    shj_kdtree.cpp \
    generateneighborxml.cpp \
    gmm.cpp \
    kmeans.cpp \
    readsuperpixeldat.cpp \
    ../../桌面/SLICSuperpixels_VC2008_SLICO_15Jun2013/SLICSuperpixels/SLIC.cpp


HEADERS += \
    watershedprocessor.h \
    thresholdprocessor.h \
    nnhdprocessor.h \
    findsimilarpoint.h \
    patchmatchprocessor.h \
    mygabor.h \
    utils.h \
    grabcut.h \
    graphcut.h \
    gco-v3.0/block.h \
    gco-v3.0/energy.h \
    gco-v3.0/GCoptimization.h \
    gco-v3.0/graph.h \
    gco-v3.0/LinkedBlockList.h \
    shj_kdtree.h \
    generateneighborxml.h \
    gmm.h \
    kmeans.h \
    readsuperpixeldat.h \
    ../../桌面/SLICSuperpixels_VC2008_SLICO_15Jun2013/SLICSuperpixels/SLIC.h


LIBS += /usr/local/lib/libopencv_core.so    \
		/usr/local/lib/libopencv_imgproc.so \
		/usr/local/lib/libopencv_highgui.so \
		/usr/local/lib/libopencv_objdetect.so \
		/usr/local/lib/libopencv_video.so	\
		/usr/local/lib/libopencv_videoio.so	\
		/usr/local/lib/libopencv_imgcodecs.so	\
		/usr/local/lib/libopencv_flann.so		\
