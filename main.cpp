

#include "watershedprocessor.h"
#include "thresholdprocessor.h"
#include "nnhdprocessor.h"
#include "findsimilarpoint.h"
#include "patchmatchprocessor.h"
#include "mygabor.h"
#include "graphcut.h"
#include "grabcut.h"
#include "shj_kdtree.h"
#include "generateneighborxml.h"
#include "kmeans.h"
#include "readsuperpixeldat.h"

int main(int argc, char *argv[])
{
    WatershedProcessor* watershedProcessor = new WatershedProcessor();
    ThresholdProcessor* thresholdProcessor = new ThresholdProcessor();
    NNHDProcessor* nNHDProcessor = new NNHDProcessor();
    FindSimilarPoint* findSimilarPoint = new FindSimilarPoint();
    PatchMatchProcessor* patchMatchProcessor = new PatchMatchProcessor();
    MyGabor* myGabor = new MyGabor();
    GCApplication* gCApplication = new GCApplication();     //grabcut
    GraphCut* graphCut = new GraphCut();    //boykov graph cut
    KMeans* kMeans = new KMeans();
    SHJ_KDTree* sHJ_KDTree = new SHJ_KDTree();
    GenerateNeighborXML* generateNeighborXML = new GenerateNeighborXML();
    ReadSuperPixelDat* readSuperPixelDat = new ReadSuperPixelDat();

    //watershedProcessor->main();
    //thresholdProcessor->main();
    //nNHDProcessor->main();
    //findSimilarPoint->main();
    //patchMatchProcessor->main();
    //myGabor->main();
    //gCApplication->main();     //grabcut
    //graphCut->main();    //boykov graph cut
    //kMeans->main();
    //sHJ_KDTree->main();
    //generateNeighborXML->main();
    readSuperPixelDat->main();
}

