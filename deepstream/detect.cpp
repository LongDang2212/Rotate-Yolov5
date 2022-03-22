#include <gst/gst.h>
#include <glib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>

#include "cuda_runtime_api.h"
#include "nvdsinfer_custom_impl.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvds_version.h"

#ifdef PLATFORM_TEGRA
#define INFER_PGIE_CONFIG_FILE "../configs/infer_aarch64.txt"
#else
#define INFER_PGIE_CONFIG_FILE "../configs/infer_x86.txt"
#endif

#define MAX_NUM_SOURCES 4
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720
#define PGIE_NET_WIDTH 640
#define PGIE_NET_HEIGHT 640

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 25000

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing NvBufSurface. */
#define MEMORY_FEATURES "memory:NVMM"

unsigned int nvds_lib_major_version = NVDS_VERSION_MAJOR;
unsigned int nvds_lib_minor_version = NVDS_VERSION_MINOR;

gint frame_number = 0;
gint g_source_id_list[MAX_NUM_SOURCES];
GMutex perf_lock;

#ifdef PLATFORM_TEGRA
GstElement *transform = NULL;
#endif

GstElement *source = NULL, *vidconv_src = NULL, *nvvidconv_src = NULL, *filter_src = NULL; // For webcam decoder
GstElement *pipeline = NULL, *streammux = NULL;
GstElement *nvvidconv = NULL, *caps_filter = NULL, *pgie = NULL, *pgie_queue = NULL,
           *sgie1 = NULL, *sgie1_queue = NULL, *sgie2 = NULL,
           *nvtracker = NULL, *nvtracker_queue = NULL,
           *nvdsanalytics = NULL, *nvdsanalytics_queue = NULL,
           *tiler = NULL, *nvosd = NULL, *tee = NULL;
// GstElement *queue1 = NULL, *v4l2_convert = NULL, *v4l2_identity = NULL, *nvsink = NULL;// *sink = NULL;
GstElement *queue1 = NULL, *sink_transform = NULL, *sink_caps_filter = NULL, *sink_encoder = NULL, *codecparse = NULL, *mux = NULL, *sink = NULL;
GstElement *queue2 = NULL, *msgconv = NULL, *msg_queue = NULL, *msgbroker = NULL;
GstElement *nvvidconv_postosd = NULL, *filter_sink = NULL, *encoder = NULL, *rtppay = NULL; // For rtsp sink

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseRYolo(std::vector<NvDsInferLayerInfo> const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
                                    NvDsInferParseDetectionParams const &detectionParams,
                                    std::vector<NvDsInferObjectDetectionInfo> &objectList);

static GstPadProbeReturn pgie_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
  GstBuffer *inbuf = GST_PAD_PROBE_INFO_BUFFER(info);

  static NvDsInferNetworkInfo networkInfo{PGIE_NET_WIDTH, PGIE_NET_HEIGHT, 3};
  NvDsInferParseDetectionParams detectionParams;
  detectionParams.perClassThreshold = {0.5};

  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta(inbuf);

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
    /* Iterate user metadata in frames to search PGIE's tensor metadata */
    for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next)
    {
      NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
      if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
        continue;

      /* convert to tensor metadata */
      NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
      for (unsigned int i = 0; i < meta->num_output_layers; i++)
      {
        NvDsInferLayerInfo *info = &meta->output_layers_info[i];
        info->buffer = meta->out_buf_ptrs_host[i];
      }
      /* Parse output tensor and fill detection results into objectList. */
      std::vector<NvDsInferLayerInfo> outputLayersInfo(meta->output_layers_info,
                                                       meta->output_layers_info + meta->num_output_layers);
      std::vector<NvDsInferObjectDetectionInfo> objectList;
#if NVDS_VERSION_MAJOR >= 5
      if (nvds_lib_major_version >= 5)
      {
        if (meta->network_info.width != networkInfo.width ||
            meta->network_info.height != networkInfo.height ||
            meta->network_info.channels != networkInfo.channels)
        {
          g_error("Failed to check pgie network info\n");
        }
      }
#endif
      NvDsInferParseRYolo(outputLayersInfo, networkInfo,
                          detectionParams, objectList);
      for (auto &obj : objectList)
      {
        NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
        obj_meta->unique_component_id = meta->unique_id;
        obj_meta->confidence = obj.detectionConfidence;
        obj_meta->object_id = UNTRACKED_OBJECT_ID;
        obj_meta->class_id = 0;

        NvDsDisplayMeta *a = nvds_acquire_display_meta_from_pool(batch_meta);
        a->num_lines = 4;
        for (int k = 0; k < a->num_lines; k++)
        {
          a->line_params[k].line_width = 3;
          a->line_params[k].line_color = (NvOSD_ColorParams){1.0, 0.0, 0.0, 1.0};
        }

        a->line_params[0].x1 = obj.box.x1 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        a->line_params[0].y1 = obj.box.y1 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        a->line_params[0].x2 = obj.box.x2 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        a->line_params[0].y2 = obj.box.y2 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;

        a->line_params[1].x1 = obj.box.x1 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        a->line_params[1].y1 = obj.box.y1 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        a->line_params[1].x2 = obj.box.x4 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        a->line_params[1].y2 = obj.box.y4 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;

        a->line_params[2].x1 = obj.box.x2 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        a->line_params[2].y1 = obj.box.y2 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        a->line_params[2].x2 = obj.box.x3 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        a->line_params[2].y2 = obj.box.y3 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;

        a->line_params[3].x1 = obj.box.x3 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        a->line_params[3].y1 = obj.box.y3 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        a->line_params[3].x2 = obj.box.x4 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        a->line_params[3].y2 = obj.box.y4 / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        nvds_add_display_meta_to_frame(frame_meta, a);

        // NvOSD_RectParams & rect_params = obj_meta->rect_params;
        // NvOSD_TextParams & text_params = obj_meta->text_params;

        /* Assign bounding box coordinates. */
        // rect_params.left = obj.left / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        // rect_params.top = obj.top / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        // rect_params.width = obj.width / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;
        // rect_params.height = obj.height / PGIE_NET_WIDTH * MUXER_OUTPUT_WIDTH;

        // /* verify bbox */
        // rect_params.left = rect_params.left * (rect_params.left>=0.0);
        // rect_params.top = rect_params.top * (rect_params.top>=0);
        // rect_params.width -= (rect_params.left + rect_params.width > MUXER_OUTPUT_WIDTH) * (rect_params.left + rect_params.width - MUXER_OUTPUT_WIDTH);
        // rect_params.height -= (rect_params.top + rect_params.height > MUXER_OUTPUT_HEIGHT) * (rect_params.top + rect_params.height - MUXER_OUTPUT_HEIGHT);

        /* Border of width 3. */
        // rect_params.border_width = 3;
        // rect_params.has_bg_color = 0;
        // rect_params.border_color = (NvOSD_ColorParams) {1, 0, 0, 1};
        // text_params.display_text = g_strdup("person");
        nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
      }
    }

    // display_meta->text_params[0].display_text = g_strdup_printf ("Total: %d", total);
    // nvds_add_display_meta_to_frame (frame_meta, display_meta);
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
  GstBuffer *inbuf = GST_PAD_PROBE_INFO_BUFFER(info);

  /*
    perf_measurement, use with:
    export NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1
    export NVDS_ENABLE_LATENCY_MEASUREMENT=1
  */
  if (nvds_enable_latency_measurement)
  {
    NvDsFrameLatencyInfo latency_info;
    g_mutex_lock(&perf_lock);
    // latency_info = (NvDsFrameLatencyInfo *)
    //   calloc(1, g_num_sources * sizeof(NvDsFrameLatencyInfo));
    auto num_sources_in_batch = nvds_measure_buffer_latency(inbuf, &latency_info);
    g_print("\n************num_sources = %d**************\n", num_sources_in_batch);
    g_mutex_unlock(&perf_lock);
  }

  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg))
  {
  case GST_MESSAGE_EOS:
    g_print("End of stream\n");
    g_main_loop_quit(loop);
    break;
  case GST_MESSAGE_ERROR:
  {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }
  default:
    break;
  }
  return TRUE;
}

static gboolean select_stream_callback(GstElement *rtspsrc, guint num,
                                       GstCaps *caps, gpointer udata)
{
  if (num != 0)
    return FALSE;
  g_print("Connected stream %d from %s \n",
          num, GST_ELEMENT_NAME(rtspsrc));

  if (gst_caps_get_size(caps) > 0)
  {
    const GstStructure *structure = gst_caps_get_structure(caps, 0);
    gchar *rstructure = gst_structure_to_string(structure);
    g_print("%s \n\n", rstructure);
  }

  return TRUE;
}

static void
cb_newpad(GstElement *decodebin, GstPad *pad, gpointer data)
{
  GstCaps *caps = gst_pad_query_caps(pad, NULL);
  const GstStructure *str = gst_caps_get_structure(caps, 0);
  const gchar *name = gst_structure_get_name(str);

  g_print("decodebin new pad %s\n", name);
  if (!strncmp(name, "video", 5))
  {
    gint source_id = (*(gint *)data);
    gchar pad_name[16] = {0};
    GstPad *sinkpad = NULL;
    g_snprintf(pad_name, 15, "sink_%u", source_id);
    sinkpad = gst_element_get_request_pad(streammux, pad_name);
    if (gst_pad_link(pad, sinkpad) != GST_PAD_LINK_OK)
    {
      g_print("Failed to link decodebin to pipeline\n");
    }
    else
    {
      g_print("Decodebin linked to pipeline\n");
    }
    gst_object_unref(sinkpad);
  }
}

static void
decodebin_child_added(GstChildProxy *child_proxy, GObject *object,
                      gchar *name, gpointer user_data)
{
  g_print("Decodebin child added: %s\n", name);
  if (g_strrstr(name, "decodebin") == name)
  {
    g_signal_connect(G_OBJECT(object), "child-added",
                     G_CALLBACK(decodebin_child_added), user_data);
  }
  else if (!strncmp(name, "source", 5))
  {
    g_object_set(G_OBJECT(object), "latency", 200, NULL);
    // g_object_set (G_OBJECT (object), "protocols", 4, NULL);
    // g_signal_connect(G_OBJECT (object), "select-stream", G_CALLBACK(select_stream_callback), NULL);
  }
}

static GstElement *
create_source_bin(guint index, gchar const *url)
{
  GstElement *bin = NULL;
  gchar bin_name[16] = {};

  g_print("INFO: Creating uridecodebin for [%s]...\n", url);
  g_snprintf(bin_name, 15, "source-bin-%02d", index);

  bin = gst_element_factory_make("uridecodebin", bin_name);
  g_object_set(G_OBJECT(bin), "uri", url, NULL);
  g_signal_connect(G_OBJECT(bin), "pad-added",
                   G_CALLBACK(cb_newpad), &g_source_id_list[index]);
  g_signal_connect(G_OBJECT(bin), "child-added",
                   G_CALLBACK(decodebin_child_added), &g_source_id_list[index]);

  return bin;
}

int main(int argc, char *argv[])
{
  g_setenv("DS_NEW_BUFAPI", "1", TRUE);

  GMainLoop *loop = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id, tiler_rows, tiler_columns;
  GstPad *pgie_src_pad = NULL, *sgie1_src_pad = NULL, *sgie2_src_pad = NULL;
  GstPad *sinkpad, *srcpad, *osd_sink_pad;

  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);
  g_mutex_init(&perf_lock);

  pipeline = gst_pipeline_new("rapid-pipeline");

  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux)
  {
    g_printerr("Init Block: One element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
               MUXER_OUTPUT_HEIGHT, "batch-size", 4,
               "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  g_object_set(G_OBJECT(streammux), "live-source", 1, NULL);

  gst_bin_add(GST_BIN(pipeline), streammux);

  GstElement *source_bin = create_source_bin(0, argv[1]);
  if (!source_bin)
  {
    g_printerr("Init Block: Failed to create source bin. Exiting.\n");
    return -1;
  }
  gst_bin_add(GST_BIN(pipeline), source_bin);

  /* Use convertor to convert from NV12 to RGBA as required by dsexample */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");

  caps_filter = gst_element_factory_make("capsfilter", NULL);

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");

  pgie_queue = gst_element_factory_make("queue", NULL);

  /* Use nvtiler to stitch o/p from upstream components */
  tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

  /* comment to dev */
  // if (!nvvidconv || !caps_filter || !pgie || !pgie_queue || !sgie1 || !sgie1_queue || !sgie2 || !tiler || !nvosd) {
  //   g_printerr ("Block DS Main: One element could not be created. Exiting.\n");
  //   return -1;
  // }

#ifndef PLATFORM_TEGRA
  /* Set properties of the nvvideoconvert element
   * requires unified cuda memory for opencv blurring on CPU
   */
  g_object_set(G_OBJECT(nvvidconv), "nvbuf-memory-type", 3, NULL);
#endif

  /* Set properties of the caps_filter element */
  GstCaps *caps =
      gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGBA", NULL);
  GstCapsFeatures *feature = gst_caps_features_new(MEMORY_FEATURES, NULL);
  gst_caps_set_features(caps, 0, feature);
  g_object_set(G_OBJECT(caps_filter), "caps", caps, NULL);

  /* Set all the necessary properties of the nvinfer element */
  g_object_set(G_OBJECT(pgie), "config-file-path", INFER_PGIE_CONFIG_FILE, NULL);

  // tiler_rows = (guint) sqrt (num_sources);
  // tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  tiler_rows = 1;
  tiler_columns = 1;
  /* we set the osd properties here */
  g_object_set(G_OBJECT(tiler), "rows", tiler_rows, "columns", tiler_columns,
               "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

  gst_bin_add_many(GST_BIN(pipeline), nvvidconv, caps_filter, pgie, pgie_queue, tiler, nvosd, NULL);

  if (!gst_element_link_many(streammux, nvvidconv, caps_filter, pgie, pgie_queue, tiler, nvosd, NULL))
  {
    g_printerr("Block DS Main: Elements could not be linked. Exiting.\n");
    return -1;
  }

  /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
  transform = gst_element_factory_make("nvegltransform", "nvegl-transform");
  // transform = gst_element_factory_make ("queue", "queue");
#endif

  queue1 = gst_element_factory_make("queue", "nvtee-que1");
  sink_transform = gst_element_factory_make("nvvideoconvert", "sink_transform");
  sink_caps_filter = gst_element_factory_make("capsfilter", "sink_caps_filter");
  sink_encoder = gst_element_factory_make("x264enc", "sink_encoder");
  codecparse = gst_element_factory_make("h264parse", "codecparse");
  mux = gst_element_factory_make("qtmux", "mux");
  sink = gst_element_factory_make("filesink", "filesink");
#ifdef PLATFORM_TEGRA
  if (!transform)
  {
    g_printerr("Block Render: One tegra element could not be created. Exiting.\n");
    return -1;
  }
#endif
  if (!sink)
  {
    g_printerr("Block Render: One element could not be created. Exiting.\n");
    return -1;
  }

  /* Set properties of the caps_filter element */
  GstCaps *sink_caps = gst_caps_from_string("video/x-raw, format=I420");
  g_object_set(G_OBJECT(sink_caps_filter), "caps", sink_caps, NULL);
  /* Set properties of the sink element */
  g_object_set(G_OBJECT(sink), "location", "video1.mp4", "sync", TRUE, "async", FALSE, NULL);

#ifdef PLATFORM_TEGRA
  gst_bin_add_many(GST_BIN(pipeline), queue1, transform, sink, NULL);
  if (!gst_element_link_many(nvosd, queue1, transform, sink, NULL))
  {
    g_printerr("Block Render: Elements could not be linked. Exiting.\n");
    return -1;
  }
#else
  gst_bin_add_many(GST_BIN(pipeline), queue1, sink_transform, sink_caps_filter, sink_encoder, codecparse, mux, sink, NULL);
  if (!gst_element_link_many(nvosd, queue1, sink_transform, sink_caps_filter, sink_encoder, codecparse, mux, sink, NULL))
  {
    g_printerr("Block Render: Elements could not be linked. Exiting.\n");
    return -1;
  }
#endif

  // queue1 = gst_element_factory_make ("queue", "nvtee-que1");

  // nvsink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
  // // nvsink = gst_element_factory_make ("fakesink", "nvvideo-renderer");

  // sink = gst_element_factory_make ("fpsdisplaysink", "fps-display");

#ifdef PLATFORM_TEGRA
  if (!transform)
  {
    g_printerr("Block Render: One tegra element could not be created. Exiting.\n");
    return -1;
  }
#endif
  if (!sink)
  {
    g_printerr("Block Render: One element could not be created. Exiting.\n");
    return -1;
  }

  /* Set properties of the sink element */
  // g_object_set (G_OBJECT (sink), "sync", FALSE, NULL);
  // g_object_set (G_OBJECT (sink), "text-overlay", FALSE, "video-sink", nvsink, "sync", FALSE, NULL);

#ifdef PLATFORM_TEGRA
  gst_bin_add_many(GST_BIN(pipeline), queue1, transform, sink, NULL);

  if (!gst_element_link_many(nvosd, queue1, transform, sink, NULL))
  {
    g_printerr("Block Render: Elements could not be linked. Exiting.\n");
    return -1;
  }
#else
  // gst_bin_add_many (GST_BIN (pipeline), queue1, sink, NULL);

  // if (!gst_element_link_many (nvosd, queue1, sink, NULL)) {
  //   g_printerr ("Block Render: Elements could not be linked. Exiting.\n");
  //   return -1;
  // }
#endif

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
  if (!osd_sink_pad)
    g_print("Unable to get sink pad\n");
  else
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref(osd_sink_pad);

  /* Add probe to get informed of the meta data generated, we add probe to
   * the source pad of PGIE's next queue element, since by that time, PGIE's
   * buffer would have had got tensor metadata. */
  pgie_src_pad = gst_element_get_static_pad(pgie, "src");
  gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                    pgie_pad_buffer_probe, NULL, NULL);

  gst_object_unref(pgie_src_pad);

  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "rapid-pipeline");

  /* Set the pipeline to "playing" state */
  g_print("INFO: Playing...\n");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  g_mutex_clear(&perf_lock);

  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  // gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);

  return 0;
}
