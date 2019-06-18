/**
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distridbution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))

/* This is a sample bounding box parsing function for the sample Resnet10
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseRetinaNet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferParseObjectInfo> &objectList)
{
  static int bboxLayerIndex = -1;
  static int classesLayerIndex = -1;
  static int scoresLayerIndex = -1;
  static NvDsInferDimsCHW scoresLayerDims;
  int numDetsToParse;

  /* Find the bbox layer */
  if (bboxLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "boxes") == 0) {
        bboxLayerIndex = i;
        break;
      }
    }
    if (bboxLayerIndex == -1) {
    std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
    return false;
    }
  }

  /* Find the scores layer */
  if (scoresLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "scores") == 0) {
        scoresLayerIndex = i;
        getDimsCHWFromDims(scoresLayerDims, outputLayersInfo[i].dims);
        break;
      }
    }
    if (scoresLayerIndex == -1) {
    std::cerr << "Could not find scores layer buffer while parsing" << std::endl;
    return false;
    }
  }

  /* Find the classes layer */
  if (classesLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "classes") == 0) {
        classesLayerIndex = i;
        break;
      }
    }
    if (classesLayerIndex == -1) {
    std::cerr << "Could not find classes layer buffer while parsing" << std::endl;
    return false;
    }
  }  

  
  /* Calculate the number of detections to parse */
  numDetsToParse = scoresLayerDims.c;

  float *bboxes = (float *) outputLayersInfo[bboxLayerIndex].buffer;
  float *classes = (float *) outputLayersInfo[classesLayerIndex].buffer;
  float *scores = (float *) outputLayersInfo[scoresLayerIndex].buffer;
  
  for (int indx = 0; indx < numDetsToParse; indx++)
  {
    float outputX1 = bboxes[indx * 4];
    float outputY1 = bboxes[indx * 4 + 1];
    float outputX2 = bboxes[indx * 4 + 2];
    float outputY2 = bboxes[indx * 4 + 3];
    float this_class = classes[indx];
    float this_score = scores[indx];
    float threshold = detectionParams.perClassThreshold[this_class];
    
    if (this_score >= threshold)
    {
      NvDsInferParseObjectInfo object;
      
      object.classId = this_class;
      object.detectionConfidence = this_score;

      object.left = outputX1;
      object.top = outputY1;
      object.width = outputX2 - outputX1;
      object.height = outputY2 - outputY1;

      objectList.push_back(object);
    }
  }
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseRetinaNet);
