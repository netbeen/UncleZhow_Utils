QT += core
QT -= gui

TARGET = gco-v3
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += \
    ../example.cpp \
    ../GCoptimization.cpp \
    ../graph.cpp \
    ../LinkedBlockList.cpp \
    ../maxflow.cpp

DISTFILES += \
    ../GCO_README.TXT

HEADERS += \
    ../block.h \
    ../energy.h \
    ../GCoptimization.h \
    ../graph.h \
    ../LinkedBlockList.h

