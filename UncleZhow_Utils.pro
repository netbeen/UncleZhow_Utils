QT += core
QT -= gui

TARGET = UncleZhow_Utils
CONFIG += console
CONFIG += c++11
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    watershedprocessor.cpp \
    thresholdprocessor.cpp \
    nnhdprocessor.cpp

HEADERS += \
    watershedprocessor.h \
    thresholdprocessor.h \
    nnhdprocessor.h

LIBS += /usr/local/lib/libopencv_core.so    \
		/usr/local/lib/libopencv_imgproc.so \
		/usr/local/lib/libopencv_highgui.so \
		/usr/local/lib/libopencv_objdetect.so \
		/usr/local/lib/libopencv_video.so	\
		/usr/local/lib/libopencv_videoio.so	\
		/usr/local/lib/libopencv_imgcodecs.so	\
		/usr/local/lib/libopencv_flann.so		\
