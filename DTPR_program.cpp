
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

#include <fstream>
#include "INeuralNetwork.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    if (argc == 1) { // если в аргументах только имя программы
        cout << "no arguments!" << endl; // выводим, что нет аргументов
    }
    else {
        // иначе выводим все аргументы, которые переданы
        for (int i = 1; i < argc; i++) {
            cout << "argv[" << i << "] - " << argv[i] << endl;
        }
    }

    Mat frame_tmp;
    Mat resImg, srcImg;

    // попытка парсинга аргументов программы !!! нужно добавить дополнительные проверки и ограничения для агрументов
    string fPathIn = argv[1];
    string fPathOut = argv[2];
    string fPathNetCfg = argv[3];
    string fPathNetWeights = argv[4];
    bool separateNetFiles = true;
    int size_min = 15;
    int previousFramesCount = 6;
    int extraFramesCount = 6;
    try {
        if (stoi(argv[5]) > 0) separateNetFiles = true;
        else separateNetFiles = false;
        size_min = stoi(argv[6]);
        previousFramesCount = stoi(argv[7]);
        extraFramesCount = stoi(argv[8]);
    }
    catch (std::exception const& e) {
        // This could not be parsed into a number so an exception is thrown.
        // atoi() would return 0, which is less helpful if it could be a valid value.
        //return -1;
    }

    // захват видеопотока
    VideoCapture capture(samples::findFile(fPathIn));
    if (!capture.isOpened()) {
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }

    // создание экземпляра класса
    INeuralNetwork new_Network(fPathNetCfg, fPathNetWeights, separateNetFiles, previousFramesCount, extraFramesCount, size_min);

    while (true) {
        Mat frame;
        capture >> frame;
        new_Network.recognize(frame);
        imshow("Frame", frame);
        int keyboard = waitKey(1);
        if (keyboard == 'q' || keyboard == 27)
        {
            break;
        }
    }


    return 0;

}