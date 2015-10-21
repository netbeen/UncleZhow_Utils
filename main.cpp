

#include "watershedprocessor.h"
#include "thresholdprocessor.h"
#include "nnhdprocessor.h"

int main(int argc, char *argv[])
{

    WatershedProcessor* watershedProcessor = new WatershedProcessor();
    //watershedProcessor->main();

    ThresholdProcessor* thresholdProcessor = new ThresholdProcessor();
    //thresholdProcessor->main();

    NNHDProcessor* nNHDProcessor = new NNHDProcessor();
    nNHDProcessor->main();

}

