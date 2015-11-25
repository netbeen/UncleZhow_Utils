

#include "watershedprocessor.h"
#include "thresholdprocessor.h"
#include "nnhdprocessor.h"
#include "findsimilarpoint.h"
#include "patchmatchprocessor.h"
#include "mygabor.h"
#include "graphcut.h"
#include "grabcut.h"
#include "shj_kdtree.h"

int main(int argc, char *argv[])
{

    WatershedProcessor* watershedProcessor = new WatershedProcessor();
    //watershedProcessor->main();

    ThresholdProcessor* thresholdProcessor = new ThresholdProcessor();
    //thresholdProcessor->main();

    NNHDProcessor* nNHDProcessor = new NNHDProcessor();
    nNHDProcessor->main();

    FindSimilarPoint* findSimilarPoint = new FindSimilarPoint();
    //findSimilarPoint->main();

    PatchMatchProcessor* patchMatchProcessor = new PatchMatchProcessor();
    //patchMatchProcessor->main();

    MyGabor* myGabor = new MyGabor();
    //myGabor->main();

    GCApplication* gCApplication = new GCApplication();     //grabcut
    //gCApplication->main();

    GraphCut* graphCut = new GraphCut();
    //graphCut->main2();

    SHJ_KDTree* sHJ_KDTree = new SHJ_KDTree();
    //sHJ_KDTree->main();
}

