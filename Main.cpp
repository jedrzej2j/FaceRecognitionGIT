#include <opencv2/core/core.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "opencv2/face.hpp"
#include "opencv2/objdetect.hpp"
#include <direct.h>
#include <sys/stat.h>
#include <stdio.h>



using namespace cv;
using namespace cv::face;
using namespace std;


class Classifier {


};

class Cascade : public Classifier {
private :
	CascadeClassifier haar_cascade;

public:
	Cascade(String xmlFile) {
		haar_cascade.load(xmlFile);

	}
	void mutliScaleDetection(vector< Rect_<int> > * faces, Mat & gray) {
		haar_cascade.detectMultiScale(gray, *faces);

	}
};

class DataBase {
private: 
	Cascade * classifier ;
	vector<Mat> color_images;
	int label_iterator;
	vector<Mat> images;
	vector<int> labels;
	int face_size;

public:
	vector<Mat> * getColorImages(){
		return &color_images;
	}
	vector<Mat> * getImages() {
		return & images;
	}
	vector<int> * getLabes() {
		return & labels;
	}
	void addLabel(int & label) {
		labels.push_back(label);
	}
	void addFace(Mat &image) {
		images.push_back(image);
	}
	int getLabelIterator() {
		return label_iterator;
	}
	void incrementLabelIterator() {
		label_iterator++;
	}
	DataBase(Cascade * classifier, int face_size) :classifier(classifier),face_size(face_size),label_iterator(1) {


	}
	void read_directory(const string& directory) {
		vector<String> filenames;
		glob(directory, filenames);

		for (size_t i = 0; i < filenames.size(); ++i)
		{
			Mat src = imread(filenames[i]);
			if (i == 0)
				color_images.push_back(src); //tu do poprawy bo moga byc zle zdjecia co nie?
			if (!src.data)
				cerr << "Problem loading image!!!" << endl;
			read_image(filenames[i], label_iterator);
		}
		label_iterator++;
	}

	void read_image(const string& filename, int label) {
		Mat tmp = imread(filename, 0);

		vector< Rect_<int> > faces;
		classifier->mutliScaleDetection(&faces, tmp);
		if (faces.size()) {
			Rect face_i = faces[0];
			Mat face = tmp(face_i);

			cv::resize(face, face, Size(face_size, face_size), 1.0, 1.0, INTER_CUBIC);
			images.push_back(face);
			labels.push_back(label);
		}
	}
	void read_frame(Mat & frame, int label, bool new_person = false) {
		Mat new_frame = frame.clone();
		if (new_person)
			color_images.push_back(new_frame);
		cvtColor(new_frame, new_frame, CV_BGR2GRAY);

		vector< Rect_<int> > faces;
		//haar_cascade.detectMultiScale(new_frame, faces);
		classifier->mutliScaleDetection(&faces, new_frame);
		if (faces.size()) {
			Rect face_i = faces[0];
			Mat face = new_frame(face_i);

			cv::resize(face, face, Size(face_size, face_size), 1.0, 1.0, INTER_CUBIC);
			 
			images.push_back(face);
			labels.push_back(label);
		}
	}
	size_t getImagesSize() {
		return images.size();
	}
};


class Face_recognizer {
private:
	String face_cascade_name;
	CascadeClassifier haar_cascade;
	int prediction;
	int image_iter;
	int im_width;
	int im_height;
	bool next_person;
	int face_size;
	string box_text;
	bool enabled = false;
	string dir;
 
	VideoCapture cap;
	Ptr<FaceRecognizer> model;
	Cascade * classifier;
	DataBase *  dataBase;

public:

	~Face_recognizer() {
		delete classifier;
	}

	Face_recognizer() {
		face_cascade_name = "haarcascade_frontalface_alt.xml";
		classifier = new Cascade(face_cascade_name);
		haar_cascade.load(face_cascade_name);
		prediction = 0;
		image_iter = 0;
		next_person = true;
		box_text = format("Click C to start algorithm.");
		enabled = false;
		dir = "image_database";
		_mkdir(dir.c_str());

		model = createLBPHFaceRecognizer();
	}

	int start_camera_tracking() {
		cap.open(0);
		if (!cap.isOpened())
			return -1;
		Mat frame;
		cap >> frame;
		face_size = frame.rows / 2;
	}

	void display_output(Mat frame_input, vector< Rect_<int> > faces) {
		Mat frame = frame_input.clone();
		for (int i = 0; i < faces.size(); i++) {
			Rect face_i = faces[i];
			rectangle(frame, face_i, CV_RGB(0, 255, 0), 1);
		}

		Mat output = Mat3b(frame.rows + 0.3*frame.rows, frame.cols, Vec3b(0, 0, 0));
		frame.copyTo(output(Rect(0, 0, frame.cols, frame.rows)));
		if (enabled) {
			if (prediction) {
				Mat small_image;
				vector<Mat> color_images = *(dataBase->getColorImages());
				double small_image_proportion = 0.2 / ((double)color_images[prediction - 1].rows / (double)frame.rows);
				cv::resize(color_images[prediction - 1], small_image, cv::Size(), small_image_proportion, small_image_proportion);
				small_image.copyTo(output(Rect(frame.cols*0.5, frame.rows + 10, small_image.cols, small_image.rows)));
			}

			string option_text;
			option_text = format("P - check");
			putText(output, option_text, Point(frame.cols - 160, frame.rows + 25), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 255, 255), 2.0);
			option_text = format("ESC - leave");
			putText(output, option_text, Point(frame.cols - 160, frame.rows + 45), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 255, 255), 2.0);
			option_text = format("SPACE - add+save");
			putText(output, option_text, Point(frame.cols - 160, frame.rows + 65), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 255, 255), 2.0);
			option_text = format("T - train");
			putText(output, option_text, Point(frame.cols - 160, frame.rows + 85), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 255, 255), 2.0);
			option_text = format("N - new");
			putText(output, option_text, Point(frame.cols - 160, frame.rows + 105), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 255, 255), 2.0);
			option_text = format("C - start/stop");
			putText(output, option_text, Point(frame.cols - 160, frame.rows + 125), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 255, 255), 2.0);
			option_text = format("To switch matching mode use numpad 1,2 or 3.");
			putText(output, option_text, Point(20, frame.rows + 125), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 255, 255), 2.0);
		}
		putText(output, box_text, Point(frame.cols*0.5 - 270, frame.rows + (output.rows - frame.rows) / 2), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 255, 255), 2.0);
		cv::imshow("camera", output);
	}

	int keboard_events(Mat frame_input, vector< Rect_<int> > faces) {
		Mat frame_copy = frame_input.clone();
		Mat gray;
		cvtColor(frame_copy, gray, CV_BGR2GRAY);

		
		string deeper_dir;
		int key = waitKey(30);
		if(!enabled)
			switch (key)
			{
			case 27://escape
				return -1;
				break;
			case 'C':
			case 'c':
				enabled = !enabled;
				if (enabled)
					box_text = format("Algorithm is working!");
				else
					box_text = format("Algorithm stopped. Use C to start again.");
				break;
			}
		else
		switch (key)
		{
		case 27://escape
			return -1;
			break;
		case 32://spacja
			dataBase->read_frame(frame_copy, dataBase->getLabelIterator(), next_person);
			if (next_person)
				next_person = false;

			deeper_dir = dir + "/" + std::to_string(dataBase->getLabelIterator());
			_mkdir(deeper_dir.c_str());
			imwrite(deeper_dir + "/" + std::to_string(image_iter) + ".jpg", frame_copy);
			image_iter++;
			box_text = format("Image has been saved.");
			break;
		case 80://p i P 
		case 112:
			cout << "Calculating new prediction ..." << endl;
			if (dataBase->getImagesSize() > 0)
				for (int i = 0; i < faces.size(); i++) {
					Rect face_i = faces[i];
					Mat face = gray(face_i);
					Mat face_resized;
					cv::resize(face, face_resized, Size(face_size, face_size), 1.0, 1.0, INTER_CUBIC);
					prediction = model->predict(face_resized);
					box_text = format("Matching object to number %d:", prediction);
				}
			break;
		case 84://t i T
		case 116:
			box_text = format("Training ...");
			model->train(*(dataBase->getImages()), *(dataBase->getLabes()));
			break;
		case 78://n i N 
		case 110:
			if (next_person == false) {
				next_person = true;
				dataBase->incrementLabelIterator();
				image_iter = 0;
			}
			box_text = format("You can add new person photos!");
			break;
		case 'C':
		case 'c':
			enabled = !enabled;
			if(enabled)
				box_text = format("Algorithm is working!");
			else
				box_text = format("Algorithm stopped. Use C to start again.");
			break;
		case 49://1
			model = createLBPHFaceRecognizer();
			model->train(*(dataBase->getImages()), *(dataBase->getLabes()));
			box_text = format("LBPHFaceRecognizer enabled.");
			break;
		case 50://2
			model = createEigenFaceRecognizer();
			model->train(*(dataBase->getImages()), *(dataBase->getLabes()));
			box_text = format("EigenFaceRecognizer enabled.");
			break;
		case 51://3
			model = createFisherFaceRecognizer();
			model->train(*(dataBase->getImages()), *(dataBase->getLabes()));
			box_text = format("FisherFaceRecognizer enabled.");
			break;
		default:
			break;
		}

	}
	 
	void track() {
		start_camera_tracking();
		fulfill_database();
		for (;;)
		{
			Mat frame;
			cap >> frame;

			vector< Rect_<int> > faces;
			Mat gray;
			cvtColor(frame, gray, CV_BGR2GRAY);
			
		// 	haar_cascade.detectMultiScale(gray, faces);
			if(enabled)
				classifier->mutliScaleDetection(&faces,gray);
			display_output(frame, faces);
			if (keboard_events(frame, faces) == -1) break;
		}
	}
 
	void fulfill_database() {
		dataBase = new DataBase(this->classifier, this->face_size);

		int i = 1;
		struct stat info;
		while (true) {
			string deeper_dir = dir + "/" + std::to_string(i);
			if (stat(deeper_dir.c_str(), &info) != 0)
				break;
			else if (info.st_mode & S_IFDIR)
				dataBase->read_directory(deeper_dir);
			else
				break;
			i++;
		}

		if (dataBase->getImagesSize() > 0)
			model->train(*(dataBase->getImages()), *(dataBase->getLabes()));
	}

};
 
 
int main() {

	Face_recognizer recognizer;
	recognizer.track();

	return 0;
}