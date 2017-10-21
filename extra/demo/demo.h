// @file demo.h
//
//  \date Created on: Oct 17, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#ifndef DEMO_H_
#define DEMO_H_
#include "darknet.h"
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define NO_IMAGE_BUFFERS  (3)

typedef enum {
  ACCEL_NEON = 0,
  ACCEL_MALI,

} TargetAccel;

typedef struct {
  float det_thr;
  int resolution;
  int batch;
  TargetAccel hw;
} DemoControl;

typedef struct {
  int cam_idx;
  double cap_fps;;
  double cap_height;
  double cap_width;
  DemoControl ctrl;

  // buffers
  //image raw_image[NO_IMAGE_BUFFERS];
  image frame[NO_IMAGE_BUFFERS];
  image resized_image[NO_IMAGE_BUFFERS];
  cv::Mat boxed_frame;

  int fb_cntr;
  int det_avg_cntr;
  int det_avg_count;

  // Demo camera
  cv::VideoCapture demo_cam;

  // detection related
  float **predictions;
  char **names;
  int no_classes;
  image **alphabet;
  box *boxes;
  float **probs;
  float *avg_activations;
  int no_detections;

  // processing threads
  pthread_t gpu_det_thread;
  pthread_t neon_det_thread;
  pthread_t image_read_thread;

  // networks
  network net_cpu;
  network net_gpu;
} DemoCtx;

extern DemoCtx gDemoCtx;
extern const char *gDemoPanelName;
extern const int gImageResMax;
extern pthread_mutex_t gDemoCtxLock;
bool InitDemoPanel();
void ResetUserCtrlVals();
bool DemoSetup();
bool ReleaseDemoCtx();
bool ReallocateDetectionBuffers(int b, int n, int w, int h, int no_outputs);
void *PreprocessThreadHandler(void *arg);
void *DetectorThreadHandler(void *arg);
void *DisplayThreadHandler(void *arg);
#endif  // DEMO_H_
