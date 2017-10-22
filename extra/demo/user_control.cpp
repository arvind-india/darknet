// @file user_control.cpp
//
//  \date Created on: Oct 16, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//
#include "demo.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>

const char *gDemoPanelName = "FaceDemo";
const char *gResolutionCtrl = "Image resolution";
const char *gDetectionThrCtrl = "Detection threshold";

const int gDetThrMax = 100;
const int gDetThrRst = 25;
const int gDetThrMin = 10;
const int gImageResMax = 640;
const int gImageResMin = 288;
const int gImageResRst = 416;

int gDetThr;
int gImageRes;

void ResetUserCtrlVals() {
  gDemoCtx.ctrl.det_thr = (float)gDetThrRst / gDetThrMax;
  gDemoCtx.ctrl.resolution = gImageResRst;
  gDemoCtx.ctrl.batch = 1;
  gDemoCtx.ctrl.hw = ACCEL_NEON;
}

void DetThrCallback(int, void*) {
  pthread_mutex_lock(&gDemoCtxLock);
  float new_thr;

  if (gDetThr < gDetThrMin) {
    new_thr = (float)gDetThrMin / gDetThrMax;
  } else {
    new_thr = (float)gDetThr / gDetThrMax;
  }
  gDemoCtx.ctrl.det_thr = new_thr;
  printf("Changing detection threshold to %.2f\n", gDemoCtx.ctrl.det_thr);
  pthread_mutex_unlock(&gDemoCtxLock);
}

void NetResCallback(int, void*) {
  pthread_mutex_lock(&gDemoCtxLock);

  int new_res = (gImageRes / 32 + 1) * 32;
  if (new_res < gImageResMin) {
    new_res = gImageResMin;
  } else if (new_res > gImageResMax) {
    new_res = gImageResMax;
  }
  if (gDemoCtx.ctrl.resolution != new_res) {
    gDemoCtx.ctrl.resolution = new_res;
    printf("Changing input image resolution to %d\n", gDemoCtx.ctrl.resolution);
  }

  pthread_mutex_unlock(&gDemoCtxLock);
}

void my_button_cb(int state, void* userdata) {
  printf("State = %d\n", state);

}

bool InitDemoPanel() {
  // create demo panel
  cv::namedWindow(gDemoPanelName, cv::WINDOW_NORMAL);

  // Trackbar for detection threshold
  cv::createTrackbar(gDetectionThrCtrl, gDemoPanelName, &gDetThr,
                     gDetThrMax, DetThrCallback);

  // Trackbar for image resolution
  cv::createTrackbar(gResolutionCtrl, gDemoPanelName, &gImageRes,
                     gImageResMax, NetResCallback);

  // this does not showup in the GUI.
  //cv::createButton("button5",my_button_cb,NULL, CV_PUSH_BUTTON, 0);
  return true;
}


