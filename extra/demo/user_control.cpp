// @file user_control.cpp
//
//  \date Created on: Oct 16, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "demo.h"


const char *gDemoPanelName = "FaceDemo";
const char *gResolutionCtrl = "Image resolution";
const char *gDetectionThrCtrl = "Detection threshold";

const int gDetThrMax = 100;
const int gDetThrRst = 25;
const int gImageResMax = 640;
const int gImageResRst = 416;

int gDetThr;
int gImageRes;

void ResetUserCtrlVals() {
  gDemoCtx.ctrl.det_thr = (float)gDetThrRst / gDetThrMax;
  gDemoCtx.ctrl.resolution = gImageResRst;
  gDemoCtx.ctrl.batch = 1;
  gDemoCtx.ctrl.hw = ACCEL_NEON;
}

void DemoControlCallback(int, void*) {
  pthread_mutex_lock(&gDemoCtxLock);
  gDemoCtx.ctrl.det_thr = (float)gDetThr / gDetThrMax;
  int new_res = (gImageRes / 32 + 1) * 32;
  if (gDemoCtx.ctrl.resolution != new_res) {
    gDemoCtx.ctrl.resolution = new_res;
    printf("Changing input image resolution to %d\n", gDemoCtx.ctrl.resolution);
  }

  printf("Changing detection threshold to %.2f\n", gDemoCtx.ctrl.det_thr);

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
                     gDetThrMax, DemoControlCallback);

  // Trackbar for image resolution
  cv::createTrackbar(gResolutionCtrl, gDemoPanelName, &gImageRes,
                     gImageResMax, DemoControlCallback);

  //cv::createButton("dummy_button", my_button_cb, NULL, QT_PUSH_BUTTON, 0);
  cv::createButton("button5",my_button_cb,NULL, cv::QT_RADIOBOX, 0);
  return true;
}


