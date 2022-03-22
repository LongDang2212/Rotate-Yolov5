#include <cstring>
#include <vector>
#include <iostream>
#include <algorithm>
#include <initializer_list>
#include "nvdsinfer_custom_impl.h"
#include <gst/gst.h>
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

  for (int indx = 0; indx < numDetsToParse; indx++)
  {
    float6 pts = bboxes[indx];
    float this_class = 0.0f;
    float this_score = scores[indx];
    float threshold = detectionParams.perClassThreshold[this_class];
    NvDsInferParseObjectInfo object;

    if (this_score >= 0.5)
    {

      object.classId = this_class;
      object.detectionConfidence = this_score;

      float cx = (pts.x1 + pts.x2) / 2;
      float cy = (pts.y1 + pts.y2) / 2;
      float w = pts.x2 - pts.x1;
      float h = pts.y2 - pts.y1;
      float x0 = -w / 2.0f;
      float x1 = w / 2.0f;
      float y0 = -h / 2.0f;
      float y1 = h / 2.0f;

      float xyxyxyxy[4][2] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
      float R[2][2] = {{pts.c, pts.s}, {pts.s, pts.c}};
      float temp[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
      for (int m = 0; m < 4; ++m)
        for (int j = 0; j < 2; ++j)
        {
          for (int k = 0; k < 2; ++k)
          {
            temp[m][j] += xyxyxyxy[m][k] * R[k][j];
          }
        }
      for (int m = 0; m < 4; ++m)
      {
        temp[m][0] += cx;
        temp[m][1] += cy;
      }
      object.box.x1 = temp[3][0];
      object.box.y1 = temp[0][1];
      object.box.x2 = temp[2][0];
      object.box.y2 = temp[1][1];
      object.box.x3 = temp[1][0];
      object.box.y3 = temp[2][1];
      object.box.x4 = temp[0][0];
      object.box.y4 = temp[3][1];

      objectList.push_back(object);
    }
  }
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseRYolo);
