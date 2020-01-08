#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>

#include "../../csrc/engine.h"

using namespace std;

// Sample program to build a TensorRT Engine from an ONNX model from RetinaNet
//
// By default TensorRT will target FP16 precision (supported on Pascal, Volta, and Turing GPUs)
//
// You can optionally provide an INT8CalibrationTable file created during RetinaNet INT8 calibration
// to build a TensorRT engine with INT8 precision

int main(int argc, char *argv[]) {
	if (argc != 3 && argc != 4) {
		cerr << "Usage: " << argv[0] << " core_model.onnx engine.plan {Int8CalibrationTable}" << endl;
		return 1;
	}

	ifstream onnxFile;
	onnxFile.open(argv[1], ios::in | ios::binary); 

	if (!onnxFile.good()) {
		cerr << "\nERROR: Unable to read specified ONNX model " << argv[1] << endl;
		return -1;
	}

	onnxFile.seekg (0, onnxFile.end);
	size_t size = onnxFile.tellg();
	onnxFile.seekg (0, onnxFile.beg);

	auto *buffer = new char[size];
	onnxFile.read(buffer, size);
	onnxFile.close();

	// Define default RetinaNet parameters to use for TRT export
	int batch = 1;
	float score_thresh = 0.05f;
	int top_n = 1000;
	size_t workspace_size =(1ULL << 30);
	float nms_thresh = 0.5;
	int detections_per_im = 100;
	bool verbose = false;
	vector<vector<float>> anchors = {
		{-12.0, -12.0, 19.0, 19.0, -8.0, -20.0, 15.0, 27.0, -18.0, -8.0, 25.0, 15.0, -16.15, -16.15, 23.15, 23.15, -11.11, -26.23, 18.11, 33.23, -23.71, -11.11, 30.71, 18.11, -21.39, -21.39, 28.39, 28.39, -15.04, -34.09, 22.04, 41.09, -30.92, -15.04, 37.92, 22.04},
		{-24.0, -24.0, 39.0, 39.0, -14.0, -36.0, 29.0, 51.0, -38.0, -16.0, 53.0, 31.0, -32.31, -32.31, 47.31, 47.31, -19.71, -47.43, 34.71, 62.43, -49.95, -22.23, 64.95, 37.23, -42.79, -42.79, 57.79, 57.79, -26.92, -61.84, 41.92, 76.84, -65.02, -30.09, 80.02, 45.09},
		{-48.0, -48.0, 79.0, 79.0, -30.0, -76.0, 61.0, 107.0, -74.0, -28.0, 105.0, 59.0, -64.63, -64.63, 95.63, 95.63, -41.95, -99.91, 72.95, 130.91, -97.39, -39.43, 128.39, 70.43, -85.59, -85.59, 116.59, 116.59, -57.02, -130.04, 88.02, 161.04, -126.86, -53.84, 157.86, 84.84}, 
		{-96.0, -96.0, 159.0, 159.0, -58.0, -148.0, 121.0, 211.0, -150.0, -60.0, 213.0, 123.0, -129.26, -129.26, 192.26, 192.26, -81.39, -194.78, 144.39, 257.78, -197.30, -83.91, 260.30, 146.91, -171.18, -171.18, 234.18, 234.18, -110.86, -253.73, 173.86, 316.73, -256.90, -114.04, 319.90, 177.04},
		{-192.0, -192.0, 319.0, 319.0, -118.0, -300.0, 245.0, 427.0, -298.0, -116.0, 425.0, 243.0, -258.53, -258.53, 385.53, 385.53, -165.30, -394.61, 292.30, 521.61, -392.09, -162.78, 519.09, 289.78, -342.37, -342.37, 469.37, 469.37, -224.90, -513.81, 351.90, 640.81, -510.63, -221.73, 637.63, 348.73}
	};

	// For now, assume we have already done calibration elsewhere 
	// if we want to create an INT8 TensorRT engine, so no need 
	// to provide calibration files or model name
	const vector<string> calibration_files;
	string model_name = "";
	string calibration_table = argc == 4 ? string(argv[3]) : "";

	// Use FP16 precision by default, use INT8 if calibration table is provided
	string precision = "FP16";
	if (argc == 4)
		precision = "INT8";

	cout << "Building engine..." << endl;
	auto engine = retinanet::Engine(buffer, size, batch, precision, score_thresh, top_n,
		anchors, nms_thresh, detections_per_im, calibration_files, model_name, calibration_table, verbose, workspace_size);
	engine.save(string(argv[2]));

	delete [] buffer;

	return 0;
}
