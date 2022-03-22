#include <vector>
#include <iostream>
#include <fstream>

#include "engine.h"

using namespace std;

int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		cerr << "Usage: " << argv[0] << " core_model.onnx engine.plan" << endl;
	}

	ifstream onnxFile;
	onnxFile.open(argv[1], ios::in | ios::binary);
	cout << "Load model from " << argv[1] << endl;

	if (!onnxFile.good())
	{
		cerr << "\nERROR: Unable to read specified ONNX model " << argv[1] << endl;
		return -1;
	}

	onnxFile.seekg(0, onnxFile.end);
	size_t size = onnxFile.tellg();
	onnxFile.seekg(0, onnxFile.beg);

	auto *buffer = new char[size];
	onnxFile.read(buffer, size);
	onnxFile.close();

	bool verbose = true;
	size_t workspace_size = (1ULL << 30);
	const vector<int> dynamic_batch_opts{1, 8, 16};

	// decode params
	float score_thresh = 0.3f;
	int top_n = 150;
	vector<vector<float>> anchors;
	anchors = {{27,  26,  20,  40,  44,  19,  34,  34,  25,  47},
						 {55,  24,  44,  38,  31,  61,  50,  50,  63,  45},
						 { 65,  62,  88,  60,  84,  79, 113,  85, 148, 122}};
	vector<float> strides;
	strides = {8, 16, 32};

	// nms params
	float nms_thresh = 0.5;
	int detections_per_im = 50;

	cout << "Building engine..." << endl;
	auto engine = ryolo::Engine(buffer, size, dynamic_batch_opts,
															score_thresh, top_n, anchors, strides,
															nms_thresh, detections_per_im,
															verbose, workspace_size);
	engine.save(string(argv[2]));

	delete[] buffer;
	return 0;
}