// @file demo_main.cpp
//
//  \date Created on: Oct 16, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//
#include <iostream>
#include <signal.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "demo.h"
#include <pthread.h>

bool gExitFlag = false;
pthread_mutex_t gDemoCtxLock;
double start_time;

void ExitDemo(int signal) {
  if (!ReleaseDemoCtx()) {
    printf("Failed to release demo context\n");
  }
  gExitFlag = true;
}

bool StartDemo() {
  cv::Mat start_screen = cv::imread("./data/crowd.png");
  if (!start_screen.data) {
    printf("Failed to load start screen\n");
    return false;
  }
  cv::imshow(gDemoPanelName, start_screen);
  cv::Mat frame;
  while (1) {
    if (!gExitFlag) {
      if (!gDemoCtx.demo_cam.read(frame)) {
        printf("Failed to read the frame from camera\n");
      }
      cv::imshow(gDemoPanelName, frame);
      cv::waitKey(5);

    }
  }
  return true;
}

bool RunDemo() {
  while (!gExitFlag) {
    start_time = what_time_is_it_now();
    if (pthread_create(&gDemoCtx.image_read_thread, 0,
                       PreprocessThreadHandler, 0)) {
      error("Thread creation failed");
    }
    if(pthread_create(&gDemoCtx.neon_det_thread, 0,
                      DetectorThreadHandler, 0)) {
      error("Thread creation failed");
    }

    DisplayThreadHandler(NULL);
    pthread_join(gDemoCtx.image_read_thread, 0);
    pthread_join(gDemoCtx.neon_det_thread, 0);
    float fps = 1 / (what_time_is_it_now() - start_time);
    printf("FPS = %.2f\n", fps);
    gDemoCtx.fb_cntr = (gDemoCtx.fb_cntr + 1) % NO_IMAGE_BUFFERS;
  }
  return true;
}

int main(int argc, char **argv) {
  std::cout << "Face and Body Detection Demo" << std::endl;

  if (!DemoSetup()) {
    printf("Demo initilization failed\n");
    return -1;
  }
  printf("Demo setup done\n");
  signal(SIGINT, ExitDemo);
  //return 0;
  if (!RunDemo()) {
    printf("Failed to start demo :(\n");
    return -1;
  }
  std::cout << "Exiting Demo Application..." << std::endl;
  return 0;
}
