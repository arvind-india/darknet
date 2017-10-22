// @file demo_threads.cpp
//
//  \date Created on: Oct 18, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//

#include "demo.h"
#include "darknet.h"

extern "C" {
extern void letterbox_image_into(image im, int w, int h, image boxed);
void mean_arrays(float **a, int n, int els, float *avg);
}

void *DetectorThreadHandler(void *arg) {
  float nms = .4;
  pthread_mutex_lock(&gDemoCtxLock);
  int new_res = gDemoCtx.ctrl.resolution;
  float det_thr = gDemoCtx.ctrl.det_thr;
  int old_res = gDemoCtx.net_cpu.h;
  if (new_res != old_res) {
    // resize network
    UpdateNetResolution(new_res, new_res);
    // FIXME : need to discard previous activations
  }
  pthread_mutex_unlock(&gDemoCtxLock);
  printf("Resolution : %dx%d\t Thr = %.2f\n", gDemoCtx.net_cpu.w,
         gDemoCtx.net_cpu.h, det_thr);
  layer l = gDemoCtx.net_cpu.layers[gDemoCtx.net_cpu.n - 1];
  float *X = gDemoCtx.resized_image[(gDemoCtx.fb_cntr+2)%NO_IMAGE_BUFFERS].data;
  float *prediction = network_predict(gDemoCtx.net_cpu, X);

  memcpy(gDemoCtx.predictions[gDemoCtx.det_avg_cntr],
         prediction, l.outputs*sizeof(float));
  mean_arrays(gDemoCtx.predictions, gDemoCtx.det_avg_count, l.outputs,
              gDemoCtx.avg_activations);
  l.output = gDemoCtx.avg_activations;

  if(l.type == DETECTION){
      get_detection_boxes(l, 1, 1, det_thr, gDemoCtx.probs,
                          gDemoCtx.boxes, 0);
  } else if (l.type == REGION){
      get_region_boxes(l, gDemoCtx.frame[0].w, gDemoCtx.frame[0].h,
                       gDemoCtx.net_cpu.w, gDemoCtx.net_cpu.h,
                       det_thr, gDemoCtx.probs, gDemoCtx.boxes,
                       0, 0, 0, 0.5, 1);
  } else {
      error("Last layer must produce detections\n");
  }
  if (nms > 0) do_nms_obj(gDemoCtx.boxes, gDemoCtx.probs, l.w*l.h*l.n,
                          l.classes, nms);

  printf("Objects:\n\n");
  image display = gDemoCtx.frame[(gDemoCtx.fb_cntr + 2) % NO_IMAGE_BUFFERS];
  draw_detections(display, gDemoCtx.no_detections, det_thr,
                  gDemoCtx.boxes, gDemoCtx.probs, 0,
                  gDemoCtx.names, gDemoCtx.alphabet, gDemoCtx.no_classes);

  gDemoCtx.det_avg_cntr = (gDemoCtx.det_avg_cntr + 1) % gDemoCtx.det_avg_count;
  return 0;
}


void *PreprocessThreadHandler(void *arg) {

  pthread_mutex_lock(&gDemoCtxLock);
  int net_h = gDemoCtx.net_cpu.h;
  int net_w = gDemoCtx.net_cpu.w;
  int fb_no = gDemoCtx.fb_cntr;
  image cur_resized = gDemoCtx.resized_image[fb_no];
  pthread_mutex_unlock(&gDemoCtxLock);
  cv::Mat bgr_frame;
  image cur_frame = gDemoCtx.frame[fb_no];


  if (!gDemoCtx.demo_cam.read(bgr_frame)) {
    printf("Failed to read the frame from camera\n");
  }

  // read form mat and normalize
  int w = bgr_frame.cols;
  int h = bgr_frame.rows;
  int step = bgr_frame.step;
  int c = bgr_frame.channels();
  unsigned char *data = (unsigned char *)bgr_frame.data;

  int i, j, k;
  for(i = 0; i < h; ++i){
      for(k= 0; k < c; ++k){
          for(j = 0; j < w; ++j){
             cur_frame.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
          }
      }
  }
  rgbgr_image(cur_frame);
  letterbox_image_into(cur_frame, net_w, net_h, cur_resized);
  return 0;
}

void DisplayImageAsMat(image im, cv::Mat &mat) {

  if(im.c == 3) rgbgr_image(im);
  int step = mat.step;
  for(int y = 0; y < im.h; ++y){
    for(int x = 0; x < im.w; ++x){
      for(int k = 0; k < im.c; ++k){
        float pixel = im.data[k*im.h*im.w + y*im.w + x] * 255;
        mat.data[y*step + x*im.c + k] = (unsigned char)pixel;
      }
    }
  }
  cv::imshow(gDemoPanelName, mat);
  cv::waitKey(1);
}

void *DisplayThreadHandler(void *arg) {
  image disp_buffer = gDemoCtx.frame[(gDemoCtx.fb_cntr + 1)%NO_IMAGE_BUFFERS];
  DisplayImageAsMat(disp_buffer, gDemoCtx.boxed_frame);
  return 0;
}
