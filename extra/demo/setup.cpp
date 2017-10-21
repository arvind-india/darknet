// @file setup.cpp
//
//  \date Created on: Oct 17, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//
#include "demo.h"
#include "darknet.h"
#include <cstdio>
#include <cassert>

DemoCtx gDemoCtx;
char *gCpuNetCfg = "../../cfg/tiny-yolo-voc.cfg";
char *gGpuNetCfg = "../../cfg/tiny-yolo-voc.cfg";
char *gNetModel = "../../models/tiny-yolo-voc.weights";
char *gClassLabelFile = "../../data/voc.names";

static bool AllocateDetectionBuffers(int b, int n, int w, int h, int no_outputs) {
  gDemoCtx.avg_activations = (float *)malloc(no_outputs * sizeof(float));
  gDemoCtx.predictions = (float **) calloc(gDemoCtx.det_avg_count, sizeof(float*));
  for(int f = 0; f < gDemoCtx.det_avg_count; ++f) {
    gDemoCtx.predictions[f] = (float *) calloc(no_outputs, sizeof(float));
  }

  gDemoCtx.boxes = (box *)calloc(w * h * n, sizeof(box));
  gDemoCtx.probs = (float **)calloc(w * h * n, sizeof(float *));
  for(int b = 0; b < w * h* n; ++b) {
    gDemoCtx.probs[b] = (float *)calloc(gDemoCtx.no_classes+1, sizeof(float));
  }
  return true;
}

bool ReallocateDetectionBuffers(int b, int n, int w, int h, int no_outputs) {
  gDemoCtx.avg_activations = (float *)realloc(gDemoCtx.avg_activations,
                                              no_outputs * sizeof(float));
  for(int f = 0; f < gDemoCtx.det_avg_count; ++f) {
    // FIXME : need memset to 0 as the original allocation used calloc
    gDemoCtx.predictions[f] = (float *) realloc(gDemoCtx.predictions[f],
                                                no_outputs * sizeof(float));
  }

  gDemoCtx.boxes = (box *)realloc(gDemoCtx.boxes, w * h * n * sizeof(box));
  gDemoCtx.probs = (float **)realloc(gDemoCtx.probs, w * h * n * sizeof(float *));
  for(int b = 0; b < w * h* n; ++b) {
    gDemoCtx.probs[b] = (float *)realloc(gDemoCtx.probs[b],
                                         (gDemoCtx.no_classes+1) * sizeof(float));
  }
  return true;
}

bool DemoCtxInit() {
  gDemoCtx.cap_height = 480;
  gDemoCtx.cap_width = 640;
  gDemoCtx.cap_fps = 2;
  gDemoCtx.cam_idx = 0;
  gDemoCtx.no_classes = 20;
  // default values for user control
  ResetUserCtrlVals();

  gDemoCtx.fb_cntr = 0;
  gDemoCtx.det_avg_cntr = 0;
  gDemoCtx.det_avg_count = 3;
  // setup all buffers, labels
  gDemoCtx.alphabet = load_alphabet();
  gDemoCtx.names = get_labels(gClassLabelFile);



  return true;
}

bool DemoNetInit() {
  gDemoCtx.net_cpu = parse_network_cfg(gCpuNetCfg);
  //gDemoCtx.net_gpu = parse_network_cfg(gGpuNetCfg);

  load_weights(&gDemoCtx.net_cpu, gNetModel);
  //load_weights(&gDemoCtx.net_gpu, gNetModel);

  set_batch_network(&gDemoCtx.net_cpu, 1);
  //set_batch_network(&gDemoCtx.net_gpu, 1);

  return true;
}



// Update network input resolution
bool UpdateNetResolution(int batch, int new_h, int new_w) {
  assert(batch == 1);
  set_batch_network(&gDemoCtx.net_cpu, batch);
  // resize network
  resize_network(&gDemoCtx.net_cpu, new_w, new_h);

  layer l = gDemoCtx.net_cpu.layers[gDemoCtx.net_cpu.n - 1];
  // TODO : assmed batch = 1
  gDemoCtx.no_detections = l.n * l.h * l.w;
  ReallocateDetectionBuffers(gDemoCtx.net_cpu.batch, l.n, l.w, l.h, l.outputs);

  return true;
}

void PrintCameraProperties() {
  printf("-------Demo Camera Info---------\n"
      "Capture resolution = %f X %f\n"
      "FPS = %f\n"
      "----------------------------------\n",
         gDemoCtx.demo_cam.get(CV_CAP_PROP_FRAME_WIDTH),
         gDemoCtx.demo_cam.get(CV_CAP_PROP_FRAME_HEIGHT),
         gDemoCtx.demo_cam.get(CV_CAP_PROP_FPS));

}

bool DemoCameraInit() {
  gDemoCtx.demo_cam.open(gDemoCtx.cam_idx);
  if (!gDemoCtx.demo_cam.isOpened()) {
    printf("Failed to open video camera with index %d\n", gDemoCtx.cam_idx);
    return false;
  }
  // set camera properties
  if (gDemoCtx.cap_height) {
    gDemoCtx.demo_cam.set(CV_CAP_PROP_FRAME_HEIGHT, gDemoCtx.cap_height);
  }
  if (gDemoCtx.cap_width) {
    gDemoCtx.demo_cam.set(CV_CAP_PROP_FRAME_WIDTH, gDemoCtx.cap_width);
  }
  if (gDemoCtx.cap_fps) {
    printf("Setting camera FPS to %f\n", gDemoCtx.cap_fps);
    if (!gDemoCtx.demo_cam.set(CV_CAP_PROP_FPS, gDemoCtx.cap_fps)) {
      printf("Failed to set target FPS\n");
    }
  }
  PrintCameraProperties();
  printf("Camera setup done\n");
  return true;
}

bool DemoBufferInit() {
  // allocate a mat for the output image
  cv::Mat temp;
  gDemoCtx.demo_cam.read(temp);
  gDemoCtx.boxed_frame = cv::Mat(temp.rows, temp.cols, CV_8UC3);

  // allocate frame buffers with max size so that we dont have to reallocate
  // when network input resolution is changed.
  for (int b = 0; b < NO_IMAGE_BUFFERS; b++) {
    gDemoCtx.frame[b] = make_image(temp.cols, temp.rows, 3);
    gDemoCtx.resized_image[b] = make_image(gImageResMax, gImageResMax, 3);
    gDemoCtx.resized_image[b].h = gDemoCtx.net_cpu.h;
    gDemoCtx.resized_image[b].w = gDemoCtx.net_cpu.w;
  }

  layer l = gDemoCtx.net_cpu.layers[gDemoCtx.net_cpu.n - 1];
  gDemoCtx.no_detections = l.n * l.h * l.w;
  AllocateDetectionBuffers(gDemoCtx.net_cpu.batch, l.n, l.w, l.h, l.outputs);

  return true;
}

bool DemoSetup() {
  if(!DemoCtxInit()) {
    printf("Demo context init failed\n");
    return false;
  }

  if (!InitDemoPanel()) {
    printf("Failed to create demo display\n");
    return false;
  }

  // create detector networks and init context
  if (!DemoNetInit()) {
    printf("Failed init networks\n");
    return false;
  }

  if (!DemoCameraInit()) {
    printf("Failed to init demo camera\n");
    return false;
  }
  if (!DemoBufferInit()) {
    printf("Demo buffer init failed\n");
    return false;
  }
  return true;
}

bool ReleaseDemoCtx() {
  gDemoCtx.demo_cam.release();

  cv::destroyAllWindows();

  free_network(gDemoCtx.net_cpu);
  free_network(gDemoCtx.net_gpu);

  // TODO:
  // release alphabet, names, probs, predictions, avg_activations

  // free up all frame buffers
  return true;
}

