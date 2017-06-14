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

static void read_image(const string& filename, int label, vector<Mat>& images, vector<int>& labels) {
	images.push_back(imread(filename, 0)); //wczytuje jako gray
	labels.push_back(label);
}

static void read_frame(Mat frame, int label, vector<Mat>& images, vector<int>& labels, vector<Mat>& color_images, bool new_person = false) {
	Mat new_frame = frame.clone();
	if (new_person)
		color_images.push_back(new_frame);
	cvtColor(new_frame, new_frame, CV_BGR2GRAY);
	images.push_back(new_frame);
	labels.push_back(label);
}

static void read_directory(const string& directory, int& label_iterator, vector<Mat>& images, vector<int>& labels, vector<Mat>& color_images) {
	vector<String> filenames; 
	glob(directory, filenames);

	for (size_t i = 0; i < filenames.size(); ++i)
	{
		Mat src = imread(filenames[i]);
		if (i==0) 
			color_images.push_back(src);
		if (!src.data)
			cerr << "Problem loading image!!!" << endl;
		read_image(filenames[i], label_iterator, images, labels);
	}
	label_iterator++;
}


int main() {
	vector<Mat> images;
	vector<Mat> color_images;
	vector<int> labels;
	int label_iterator = 1;

/*/////////////////// BAZA WEJSCIOWA ////////////////////
\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/*/
	
	read_directory("test_faces/jedrzej", label_iterator, images, labels, color_images);
	read_directory("test_faces/iwona", label_iterator, images, labels, color_images);

/* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
/////////////////// BAZA WEJSCIOWA //////////////////// */	

	//najwazniejsza rzecz projektu ;d 
	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	model->train(images, labels);

	//tutaj jest klasyfikator specjalny, on opiera sie na xml jakims dzikim i sluzy do znajdowania pionowych twarzy na obrazie - izi
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	CascadeClassifier haar_cascade;
	haar_cascade.load(face_cascade_name);

	namedWindow("camera", 1);
	int prediction = 0;
	int image_iter = 0;
	bool check_first = false;
	int im_width;
	int im_height;
	bool next_person = true;

	VideoCapture cap(0);
	if (!cap.isOpened())
		return -1;

	for (;;)
	{
		Mat frame;
		cap >> frame; 
		//tutaj za pierwszym razem okreslamy wielkosc zdjec (1 klatka z kamerki)
		if (!check_first) {
			im_width = frame.cols;
			im_height = frame.rows;
			check_first = true;
		}
		Mat frame_copy=frame.clone();

		//znajduje twarze na obrazie
		Mat gray;
		cvtColor(frame, gray, CV_BGR2GRAY);
		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(gray, faces);

		//to stawia prostokat
		for (int i = 0; i < faces.size(); i++) {
			Rect face_i = faces[i];
			rectangle(frame, face_i, CV_RGB(0, 255, 0), 1);
		}

		//dolny pasek z informacjami
		Mat output = Mat3b(frame.rows + 0.25*frame.rows, frame.cols, Vec3b(0, 0, 0));
		frame.copyTo(output(Rect(0, 0, frame.cols, frame.rows)));
		string box_text;
		if (prediction) {
			Mat small_image;
			cv::resize(color_images[prediction-1], small_image, cv::Size(), 0.2, 0.2); //prediction-1 bo prediction zaczynamy od 1, a kolorwe obrazy od 0
			small_image.copyTo(output(Rect(frame.cols*0.5, frame.rows + 10, small_image.cols, small_image.rows)));
			box_text = format("Dopasowanie do obiektu nr %d:", prediction);
		}
		else {
			box_text = format("Brak dopasowania.");
		}
		putText(output, box_text, Point(frame.cols*0.5-270, frame.rows+(output.rows-frame.rows)/2), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
		box_text = format("P - sprawdz");
		putText(output, box_text, Point(frame.cols-140, frame.rows + 15), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
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
		int key = waitKey(30);
		//escape - wyjscie
		if (key == 27) { break; } 
		//spacja - dodawanie fotki do nowej puli (lokalnie) jak chcemy dodawac kolejna osobe to klikamy N
		else if (key == 32) { 
			read_frame(frame_copy, label_iterator, images, labels, color_images, next_person);
			if (next_person)
				next_person = false;
			cout << "New photo added." << endl; 
		}
		//p i P - oblicza prawdopodobienstwo ze dana mordka jest czyjas
		else if (key == 80 || key == 112)  { 
			cout << "Calculating new prediction ..."<<endl;
			for (int i = 0; i < faces.size(); i++) {
				Rect face_i = faces[i];
				Mat face = gray(face_i);
				Mat face_resized;
				cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
				prediction = model->predict(face_resized);
			}
		}
		//s i S - zapisuje ew. zdjecie do nowej bazy
		else if (key == 83 || key == 115) {
			string dir = "new_image_dir";
			_mkdir(dir.c_str());
			imwrite(dir +"/"+ std::to_string(image_iter) + ".jpg", frame_copy);
			image_iter++;
		}
		//t i T - trenujemy klasyfikator
		else if (key == 84 || key == 116) { 
			model->train(images, labels);
			cout << "Training ..." << endl;
		}
		//n i N - po kliknieciu dodajemy kolejna osobe do lokalnych
		else if (key == 78 || key == 110) {
			next_person = true;
			label_iterator++;
			cout << "You can add new person photos!" << endl;
		}
	}

	return 0;
}