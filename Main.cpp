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
/*
//w katalogu z projektem powinien znajdowac sie haarcascade_frontalface_alt.xml
//nie jest jeszcze oprogramowane dzialanie w przypadku kilku twarzy na focie, bede musial to rozkminic, albo sie to po prostu pojebie ;d
*/


using namespace cv;
using namespace cv::face;
using namespace std;



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
	vector<Mat> images;
	vector<Mat> color_images;
	vector<int> labels;
	int label_iterator;
	VideoCapture cap;
	Ptr<FaceRecognizer> model;

public:
	Face_recognizer(){
		face_cascade_name = "haarcascade_frontalface_alt.xml";
		haar_cascade.load(face_cascade_name);
		prediction = 0;
		image_iter = 0;
		next_person = true;
		label_iterator = 1;
		model = createLBPHFaceRecognizer();
		fullfil_database();
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

		Mat output = Mat3b(frame.rows + 0.25*frame.rows, frame.cols, Vec3b(0, 0, 0));
		frame.copyTo(output(Rect(0, 0, frame.cols, frame.rows)));
		string box_text;
		if (prediction) {
			Mat small_image;
			double small_image_proportion = 0.2 / ((double)color_images[prediction - 1].rows / (double)frame.rows);
			cv::resize(color_images[prediction - 1], small_image, cv::Size(), small_image_proportion, small_image_proportion); 
			small_image.copyTo(output(Rect(frame.cols*0.5, frame.rows + 10, small_image.cols, small_image.rows)));
			box_text = format("Dopasowanie do obiektu nr %d:", prediction);
		}
		else {
			box_text = format("Brak dopasowania.");
		}

		putText(output, box_text, Point(frame.cols*0.5 - 270, frame.rows + (output.rows - frame.rows) / 2), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
		box_text = format("P - sprawdz");
		putText(output, box_text, Point(frame.cols - 140, frame.rows + 15), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
		box_text = format("ESC - wyjdz");
		putText(output, box_text, Point(frame.cols - 140, frame.rows + 35), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
		box_text = format("SPACE - dodaj");
		putText(output, box_text, Point(frame.cols - 140, frame.rows + 55), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
		box_text = format("T - trenuj");
		putText(output, box_text, Point(frame.cols - 140, frame.rows + 75), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
		box_text = format("S - zapisz");
		putText(output, box_text, Point(frame.cols - 140, frame.rows + 95), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
		box_text = format("N - nowy");
		putText(output, box_text, Point(frame.cols - 140, frame.rows + 115), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
		cv::imshow("camera", output);
	}

	int keboard_events(Mat frame_input, vector< Rect_<int> > faces) {
		Mat frame_copy = frame_input.clone();
		Mat gray;
		cvtColor(frame_copy, gray, CV_BGR2GRAY);

		string dir = "new_image_dir";
		int key = waitKey(30);
		switch (key)
		{
		case 27://escape
			return -1;
			break;
		case 32://spacja
			read_frame(frame_copy, label_iterator, next_person);
			if (next_person)
				next_person = false;
			cout << "New photo added." << endl;
			break;
		case 80://p i P 
		case 112:
			cout << "Calculating new prediction ..." << endl;
			if (images.size() > 0)
				for (int i = 0; i < faces.size(); i++) {
					Rect face_i = faces[i];
					Mat face = gray(face_i);
					Mat face_resized;
					cv::resize(face, face_resized, Size(face_size, face_size), 1.0, 1.0, INTER_CUBIC);
					prediction = model->predict(face_resized);
				}
			break;
		case 83://s i S
		case 115:
			_mkdir(dir.c_str());
			imwrite(dir + "/" + std::to_string(image_iter) + ".jpg", frame_copy);
			image_iter++;
			break;
		case 84://t i T
		case 116:
			model->train(images, labels);
			cout << "Training ..." << endl;
			break;
		case 78://n i N 
		case 110:
			next_person = true;
			label_iterator++;
			cout << "You can add new person photos!" << endl;
			break;
		default:
			break;
		}

	}



	void track() {
		start_camera_tracking();

		for (;;)
		{
			Mat frame;
			cap >> frame;

			vector< Rect_<int> > faces;
			Mat gray;
			cvtColor(frame, gray, CV_BGR2GRAY);
			haar_cascade.detectMultiScale(gray, faces);

			display_output(frame,faces);
			if(keboard_events(frame,faces)==-1) break;
		}
	}



	void read_image(const string& filename, int label) {
		Mat tmp = imread(filename, 0);

		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(tmp, faces);
		if (faces.size()) {
			Rect face_i = faces[0];
			Mat face = tmp(face_i);

			cv::resize(face, face, Size(face_size, face_size), 1.0, 1.0, INTER_CUBIC);
			images.push_back(face);
			labels.push_back(label);
		}
	}

	void read_frame(Mat frame, int label, bool new_person = false) {
		Mat new_frame = frame.clone();
		if (new_person)
			color_images.push_back(new_frame);
		cvtColor(new_frame, new_frame, CV_BGR2GRAY);

		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(new_frame, faces);

		if (faces.size()) {
			Rect face_i = faces[0];
			Mat face = new_frame(face_i);

			cv::resize(face, face, Size(face_size, face_size), 1.0, 1.0, INTER_CUBIC);
			images.push_back(face);
			labels.push_back(label);
		}
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

	void fullfil_database(){
		//read_directory("test_faces/iwona");
		//read_directory("test_faces/szwagier");
		//read_directory("test_faces/jagoda");
		//read_directory("test_faces/jedrzej2");
		//read_directory("test_faces/jedrzej");
		//read_directory("test_faces/a");
		//read_directory("test_faces/b");
		//read_directory("test_faces/c");
		if (images.size() > 0)
			model->train(images, labels);
	}

};


int main() {

	Face_recognizer recognizer;
	recognizer.track();

	return 0;
}