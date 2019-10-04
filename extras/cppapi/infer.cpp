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

#include "../../csrc/engine.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
	if (argc != 3) {
		cerr << "Usage: " << argv[0] << " engine.plan image.jpg" << endl;
		return 1;
	}

	cout << "Loading engine..." << endl;
	auto engine = retinanet::Engine(argv[1]);

	cout << "Preparing data..." << endl;
	auto image = imread(argv[2], IMREAD_COLOR);
	auto inputSize = engine.getInputSize();
	cv::resize(image, image, Size(inputSize[1], inputSize[0]));
        cv::Mat pixels;
        image.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);

        int channels = 3;
        vector<float> img;
        vector<float> data (channels * inputSize[0] * inputSize[1]);

        if (pixels.isContinuous())
            img.assign((float*)pixels.datastart, (float*)pixels.dataend);
        else {
            cerr << "Error reading image " << argv[2] << endl;
            return -1;
        }

        vector<float> mean {0.485, 0.456, 0.406};
        vector<float> std {0.229, 0.224, 0.225};
  
        for (int c = 0; c < channels; c++) {
            for (int j = 0, hw = inputSize[0] * inputSize[1]; j < hw; j++) {
                data[c * hw + j] = (img[channels * j + 2 - c] - mean[c]) / std[c];
            }
        }        

	// Create device buffers
	void *data_d, *scores_d, *boxes_d, *classes_d;
	auto num_det = engine.getMaxDetections();
	cudaMalloc(&data_d, 3 * inputSize[0] * inputSize[1] * sizeof(float));
	cudaMalloc(&scores_d, num_det * sizeof(float));
	cudaMalloc(&boxes_d, num_det * 4 * sizeof(float));
	cudaMalloc(&classes_d, num_det * sizeof(float));

	// Copy image to device
	size_t dataSize = data.size() * sizeof(float);
	cudaMemcpy(data_d, data.data(), dataSize, cudaMemcpyHostToDevice);

	// Run inference n times
	cout << "Running inference..." << endl;
	const int count = 100;
	auto start = chrono::steady_clock::now();
 	vector<void *> buffers = { data_d, scores_d, boxes_d, classes_d };
	for (int i = 0; i < count; i++) {
		engine.infer(buffers, 1);
	}
	auto stop = chrono::steady_clock::now();
	auto timing = chrono::duration_cast<chrono::duration<double>>(stop - start);
	cout << "Took " << timing.count() / count << " seconds per inference." << endl;

	// Get back the bounding boxes
	auto scores = new float[num_det];
	auto boxes = new float[num_det * 4];
	auto classes = new float[num_det];
	cudaMemcpy(scores, scores_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);
	cudaMemcpy(boxes, boxes_d, sizeof(float) * num_det * 4, cudaMemcpyDeviceToHost);
	cudaMemcpy(classes, classes_d, sizeof(float) * num_det, cudaMemcpyDeviceToHost);

	for (int i = 0; i < num_det; i++) {
		// Show results over confidence threshold
		if (scores[i] >= 0.3f) {
			float x1 = boxes[i*4+0];
			float y1 = boxes[i*4+1];
			float x2 = boxes[i*4+2];
			float y2 = boxes[i*4+3];
			cout << "Found box {" << x1 << ", " << y1 << ", " << x2 << ", " << y2
				<< "} with score " << scores[i] << " and class " << classes[i] << endl;

			// Draw bounding box on image
			cv::rectangle(image, Point(x1, y1), Point(x2, y2), cv::Scalar(0, 255, 0));
		}
	}

	delete[] scores, boxes, classes; 

	// Write image
	imwrite("detections.png", image);

	return 0;
}
