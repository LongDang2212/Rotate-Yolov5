#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <initializer_list>
#include <cmath>
#include "nvdsinfer_custom_impl.h"
#include <omp.h>
#include <gst/gst.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
// #define MIN(a,b) ((a) < (b) ? (a) : (b))
/* This is a sample bounding box parsing function for the sample Resnet10
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseRYolo(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                    NvDsInferNetworkInfo const &networkInfo,
                                    NvDsInferParseDetectionParams const &detectionParams,
                                    std::vector<NvDsInferParseObjectInfo> &objectList)
{
  static int bboxLayerIndex = -1;
  static int scoresLayerIndex = -1;
  static NvDsInferDimsCHW scoresLayerDims;
  int numDetsToParse;

  /* Find the bbox layer */
  if (bboxLayerIndex == -1)
  {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++)
    {
      if (strcmp(outputLayersInfo[i].layerName, "boxes") == 0)
      {
        bboxLayerIndex = i;
        break;
      }
    }
    if (bboxLayerIndex == -1)
    {
      std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
      return false;
    }
  }

  /* Find the scores layer */
  if (scoresLayerIndex == -1)
  {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++)
    {
      if (strcmp(outputLayersInfo[i].layerName, "scores") == 0)
      {
        scoresLayerIndex = i;
        getDimsCHWFromDims(scoresLayerDims, outputLayersInfo[i].inferDims);
        break;
      }
    }
    if (scoresLayerIndex == -1)
    {
      std::cerr << "Could not find scores layer buffer while parsing" << std::endl;
      return false;
    }
  }

  /* Calculate the number of detections to parse */
  numDetsToParse = scoresLayerDims.c;

  float6 *bboxes = (float6 *)outputLayersInfo[bboxLayerIndex].buffer;
  float *scores = (float *)outputLayersInfo[scoresLayerIndex].buffer;
  // #pragma omp parallel for
  for (int indx = 0; indx < numDetsToParse; indx++)
  {
    float6 pts = bboxes[indx];
    float this_score = scores[indx];
    NvDsInferParseObjectInfo object;

    if (this_score >= 0.9)
    {
      
      object.classId = 0;
      object.detectionConfidence = this_score;

      float cx = (pts.x1 + pts.x2) / 2;
      float cy = (pts.y1 + pts.y2) / 2;
      float w = pts.x2 - pts.x1;
      float h = pts.y2 - pts.y1;
      object.angle = atan2(pts.s, pts.c);
      object.cx = cx;
      object.cy = cy;
      object.width = w;
      object.height = h;

      objectList.push_back(object);
    }
  }
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseRYolo);
