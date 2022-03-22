#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

#include "engine.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 4)
    {
        cerr << "Usage: " << argv[0] << " engine.plan image.jpg [<OUTPUT>.png]" << endl;
        return 1;
    }

    cout << "Loading engine..." << endl;
    auto engine = std::unique_ptr<ryolo::Engine>(new ryolo::Engine(argv[1]));

    cout << "Preparing data..." << endl;
    auto image = imread(argv[2], IMREAD_COLOR);
    auto inputSize = engine->getInputSize();
    cv::resize(image, image, Size(inputSize[1], inputSize[0]));
    cv::Mat pixels;
    image.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);

    int channels = 3;
    vector<float> img;
    vector<float> data(channels * inputSize[0] * inputSize[1]);

    if (pixels.isContinuous())
        img.assign((float *)pixels.datastart, (float *)pixels.dataend);
    else
    {
        cerr << "Error reading image " << argv[2] << endl;
        return -1;
    }

    // vector<float> mean {0.485, 0.456, 0.406};
    // vector<float> std {0.229, 0.224, 0.225};
    vector<float> mean{0.0, 0.0, 0.0};
    vector<float> std{1, 1, 1};

    for (int c = 0; c < channels; c++)
    {
        for (int j = 0, hw = inputSize[0] * inputSize[1]; j < hw; j++)
        {
            data[c * hw + j] = (img[channels * j + 2 - c] - mean[c]) / std[c];
        }
    }

    // Create device buffers
    void *data_d, *scores_d, *boxes_d, *classes_d;
    auto num_det = engine->getMaxDetections();
    cudaMalloc(&data_d, 3 * inputSize[0] * inputSize[1] * sizeof(float));
    cudaMalloc(&scores_d, num_det * sizeof(float));
    cudaMalloc(&boxes_d, num_det * 6 * sizeof(float));
    cudaMalloc(&classes_d, num_det * sizeof(float));

    // Copy image to device
    size_t dataSize = data.size() * sizeof(float);
    cudaMemcpy(data_d, data.data(), dataSize, cudaMemcpyHostToDevice);

    // Run inference n times
    cout << "Running inference..." << endl;
    const int count = 1;
    auto start = chrono::steady_clock::now();
    vector<void *> buffers = {data_d, scores_d, boxes_d, classes_d};
    for (int i = 0; i < count; i++)
    {
        engine->infer(buffers, 1);
    }
    auto stop = chrono::steady_clock::now();
    auto timing = chrono::duration_cast<chrono::duration<double>>(stop - start);
    cout << "Took " << timing.count() / count << " seconds per inference." << endl;

    cudaFree(data_d);

    // Get back the bounding boxes
    unique_ptr<float[]> scores(new float[num_det]);
    unique_ptr<float[]> boxes(new float[num_det * 6]);
    unique_ptr<float[]> classes(new float[num_det]);
    cudaMemcpy(scores.get(), scores_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
    cudaMemcpy(boxes.get(), boxes_d, sizeof(float) * num_det * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(classes.get(), classes_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);

    cudaFree(scores_d);
    cudaFree(boxes_d);
    cudaFree(classes_d);

    for (int i = 0; i < num_det; i++)
    {
        // Show results over confidence threshold
        if (scores[i] >= 0.5f)
        {
            cout << scores[i] << endl;
            float xmin = boxes[i * 6 + 0];
            float ymin = boxes[i * 6 + 1];
            float xmax = boxes[i * 6 + 2];
            float ymax = boxes[i * 6 + 3];
            float sin = boxes[i * 6 + 4];
            float cos = boxes[i * 6 + 5];
            float cx = (xmin + xmax) / 2;
            float cy = (ymin + ymax) / 2;
            float w = xmax - xmin;
            float h = ymax - ymin;
            float x0 = -w / 2.0f;
            float x1 = w / 2.0f;
            float y0 = -h / 2.0f;
            float y1 = h / 2.0f;

            float xyxyxyxy[4][2] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
            float R[2][2] = {{cos, sin}, {sin, cos}};
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
            cout << "Found box with score " << scores[i] << " and class " << classes[i] << endl;

            // Draw bounding box on image
            
            cv::line(image, Point(temp[3][0], temp[0][1]), Point(temp[2][0], temp[1][1]), cv::Scalar(0, 255, 0));
            cv::line(image, Point(temp[3][0], temp[0][1]), Point(temp[0][0], temp[3][1]), cv::Scalar(0, 255, 0));
            cv::line(image, Point(temp[1][0], temp[2][1]), Point(temp[2][0], temp[1][1]), cv::Scalar(0, 255, 0));
            cv::line(image, Point(temp[1][0], temp[2][1]), Point(temp[0][0], temp[3][1]), cv::Scalar(0, 255, 0));
        }
    }

    // Write image
    string out_file = argc == 4 ? string(argv[3]) : "detections.png";
    cout << "Saving result to " << out_file << endl;
    imwrite(out_file, image);

    return 0;
}