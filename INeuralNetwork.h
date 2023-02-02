#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/video.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <chrono>
#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include <vector>

// параметры запуска нейронной сети !!! число классов и порог чувствительности, могут быть внешними параметрами программы при необходимости
const float CONFIDENCE_THRESHOLD = 0;
const float NMS_THRESHOLD = 0.4;
const int NUM_CLASSES = 3;

class INeuralNetwork
{
public:
    enum class TargetType
    {
        Undefined = 0, // неизвестная цель
        UAVTypePlane, // БПЛА тип самолет
        UAVTypeQuadcopter, // БПЛА тип квадрокоптер
        Bird // птица
    };



    using Frame = cv::Mat; // кадр

    // дополнительная информация для методов оптического потока и экстраполяции
    cv::Mat LastFrame; // предыдущий кадр
    uint prevFrameCount; // число кадров для расчета экстраполяции
    uint extraFrameCount; // число кадров, на которое делается прогноз
    std::vector <cv::Point2f> prevCoordinates; // вектор с координатами объектов в предыдущих кадрах
    std::vector<cv::Point2f> p0;

    uint size_min; // минимальный линейный размер рассматриваемого объекта

    // переменные для работы нейронной сети
    std::vector<std::string> class_names;
    cv::dnn::Net net;

    

    // colors for bounding boxes !!! сейчас работает внутри класса, можно это вынести
    std::vector<cv::Scalar> colors;

    const int NUM_COLORS = 4; // большего числа цветов пока не нужно, сами цвета тоже условные

    struct RecognitionResult
    {
        RecognitionResult(uint xInp, uint yInp, uint sizeXInp, uint sizeYInp, TargetType tTypeInp) { x = xInp; y = yInp; sizeX = sizeXInp; sizeY = sizeYInp; tType = tTypeInp; }
        // Координаты:
        // левый верхний угол - (0; 0)
        // правый нижний угол – (X_MAX; Y_MAX)
        uint x = 0; // Координата центра цели в пикселях по горизонтали
        uint y = 0; // Координата центра цели в пикселях по вертикали
        uint sizeX = 0; // Размер цели по координате X
        uint sizeY = 0; // Размер цели по координате Y
        TargetType tType{ TargetType::Undefined };
    };
public:
    INeuralNetwork(std::string netConfigPath, std::string netWeightsPath = "", bool isSeparateWeights = false, std::string fPathClasses = "classes.txt", uint previousFCount = 6, uint extrapolationFCount = 6, uint minObjSize = 10) // конструктор класса
    {
        prevFrameCount = previousFCount; // число кадров для вычисления экстраполяции
        extraFrameCount = extrapolationFCount; // число кадров на которое проводится экстраполяция, по умолчанию - 6 (0,1 секунды при скорости кадров 60 в секунду) 
        //!!! нужна поправка на реальное время, если в реальности кадры могут поступать неравномерно

        size_min = minObjSize;

        colors.push_back({ 0, 255, 255 });
        colors.push_back({ 255, 255, 0 });
        colors.push_back({ 0, 255, 0 });
        colors.push_back({ 255, 0, 0 });

        std::ifstream class_file("classes.txt"); // файл с именами классов
                                                //!!! в дальнейшем это тоже можно вынести в настраивамые поля класса 
                                                //! (или вообще убрать, если 4 наших класса будут жестко заданы внутри программы)
        if (!class_file)
        {
            std::cerr << "failed to open classes.txt\n";
            //return 0;
        }

        std::string line;
        while (std::getline(class_file, line))
            class_names.push_back(line);

        if (isSeparateWeights) // если веса представлены отдельными файлами, функция вызывается с 2 параметрами
            net = cv::dnn::readNet(netWeightsPath, netConfigPath);
        else
            net = cv::dnn::readNet(netConfigPath);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA); // !!! на CUDA протестировать не успел, в теории должно работать, но возможно нужны будут дополнительные настройки
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    };

    virtual ~INeuralNetwork() = default;

    // функция линейной интерполяции
    virtual cv::Point2f line_interpolation(std::vector <cv::Point2f> points, uint points_between, uint points_after)
    {
        int closest_point = points.size() - 1;
        int far_point = points.size() - points_between - 1; // из вектора точек берутся точки в количестве, необходимом для расчета экстраполяции

        cv::Point2f p_interp;
        p_interp.x = points[closest_point].x + (points[closest_point].x - points[far_point].x) * (float)points_after / (float)points_between;
        p_interp.y = points[closest_point].y + (points[closest_point].y - points[far_point].y) * (float)points_after / (float)points_between;
        return p_interp;
    }

    // функция распознавания нейронной сетью
    virtual std::vector<RecognitionResult> dnn_execution(Frame& currentFrame, cv::Rect ROI = cv::Rect(0,0,100,100))
    {
        std::vector<RecognitionResult> results;
        
        auto output_names = net.getUnconnectedOutLayersNames();

        cv::Mat frame, blob;
        std::vector<cv::Mat> detections;
        
        frame = currentFrame(ROI); // выделение части кадра с зафиксированными движущимися объектами

        //auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F); // размер blob зависит от конфигурации обученной сети и напрямую влияет на производительность,
                                                                                                            //!!! возможно нужно будет вынести в параметры
        net.setInput(blob);

        // часть кода для оценки времени работы нейронной сети
        //auto dnn_start = std::chrono::steady_clock::now();
        
        // вызов процедуры распознавания
        net.forward(detections, output_names);

        //auto dnn_end = std::chrono::steady_clock::now();

        std::vector<int> indices[NUM_CLASSES];
        std::vector<cv::Rect> boxes[NUM_CLASSES];
        std::vector<float> scores[NUM_CLASSES];

        for (auto& output : detections)
        {
            const auto num_boxes = output.rows;
            for (int i = 0; i < num_boxes; i++)
            {
                auto x = output.at<float>(i, 0) * frame.cols;
                auto y = output.at<float>(i, 1) * frame.rows;
                auto width = output.at<float>(i, 2) * frame.cols;
                auto height = output.at<float>(i, 3) * frame.rows;
                cv::Rect rect(x - width / 2, y - height / 2, width, height);

                for (int c = 0; c < NUM_CLASSES; c++)
                {
                    auto confidence = *output.ptr<float>(i, 5 + c);
                    if (confidence >= CONFIDENCE_THRESHOLD)
                    {
                        boxes[c].push_back(rect);
                        scores[c].push_back(confidence);
                    }
                }
            }
        }

        for (int c = 0; c < NUM_CLASSES; c++)
            cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

        
        // выбор прямоугольника с наибольшим показателем точности распознавания на выбранном участке изображения
        std::vector<float> nonZeroScores;
        std::vector<int> nonZeroScoresIdx;
        std::vector<int> nonZeroClass;

        for (int c = 0; c < NUM_CLASSES; c++)
        {
            for (size_t i = 0; i < indices[c].size(); ++i)
            {

                auto idx = indices[c][i];

                nonZeroScores.push_back(scores[c][idx]);
                nonZeroScoresIdx.push_back(idx);
                nonZeroClass.push_back(c);
            }
        }

        // если есть хотя бы один распознанный объект на изображении с достаточной точностью
        if (nonZeroScores.size() > 0)
        {
            int mostPromisingObj = std::distance(nonZeroScores.begin(), std::max_element(nonZeroScores.begin(), nonZeroScores.end()));

            const auto& rect = boxes[nonZeroClass[mostPromisingObj]][nonZeroScoresIdx[mostPromisingObj]];

            // сохранение обнаруженного объекта и передача во внешнюю функцию
            TargetType currentTargetType;
            switch (nonZeroClass[mostPromisingObj])
            {
            //case 4:
            case 2:
                currentTargetType = TargetType::UAVTypePlane;
                break;
            //case 14:
            case 0:
                currentTargetType = TargetType::Bird;
                break;
            //case 33:
            case 1:
                currentTargetType = TargetType::UAVTypeQuadcopter;
                break;
            default:
                currentTargetType = TargetType::Undefined;
                break;
            }
            RecognitionResult curResult((uint)(rect.x + ROI.x), (uint)(rect.y + ROI.y), (uint)(rect.width), (uint)(rect.height), currentTargetType);
            results.push_back(curResult);
        }

        // часть кода для оценки времени работы нейронной сети
        /*auto total_end = std::chrono::steady_clock::now();

        float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
        float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        std::ostringstream stats_ss;
        stats_ss << std::fixed << std::setprecision(2);
        stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();

        int baseline;
        auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(currentFrame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(currentFrame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));*/

        return results;
    }

    // Метод принимает на вход кадр и выдает обнаруженные цели
    virtual std::vector<RecognitionResult> recognize(Frame& currentFrame)
    {
        std::vector<RecognitionResult> resultPoints;

        cv::Mat old_gray, new_gray; // изображения в градациях серого
        std::vector<cv::Point2f> p1; // найденные точки на изображениях, предыдущем и текущем

        if (currentFrame.size() != LastFrame.size()) // если это первый вызов метода, запоминаем кадр и ждем следующий для запуска метода оптического потока
        {
            LastFrame = currentFrame.clone();
            cv::cvtColor(LastFrame, old_gray, cv::COLOR_BGR2GRAY);
            goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, cv::Mat(), 3, false, 0.04);
            return resultPoints;
        }

        // Create a mask image for drawing purposes
        cv::Mat mask = cv::Mat::zeros(currentFrame.size(), currentFrame.type());

        cv::cvtColor(LastFrame, old_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(currentFrame, new_gray, cv::COLOR_BGR2GRAY);

        if (p0.size() > 0)
        {
            // вычисление оптического потока
            std::vector<cv::Point2f> good_new;
            std::vector<uchar> status;
            cv::Mat err;
            cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 10, 0.03);
            calcOpticalFlowPyrLK(old_gray, new_gray, p0, p1, status, err, cv::Size(size_min, size_min), 3, criteria);



            int movPoints = 0;
            for (uint i = 0; i < p0.size(); i++)
            {
                // поиск значимых движущихся точек и подсчет их количества
                if (status[i] == 1) {
                    good_new.push_back(p1[i]);

                    if ((abs((p1[i] - p0[i]).x) + abs((p1[i] - p0[i]).y)) > 0.1)
                        movPoints++;
                }
            }

            // !!! поиск объектов пока что упрощен до одного объекта в кадре для целей тестирования
            // !!! условие для раздельного сопровождения объектов в оптическом потоке допишу в ближайшее время
            // поиск объекта в предыдущем кадре
            cv::Point2i p_left_top_corner, p_right_bottom_corner;
            sort(p0.begin(), p0.end(), [](cv::Point2f const& a, cv::Point2f const& b) { return a.x < b.x; });

            p_left_top_corner.x = round(p0[0].x);
            p_right_bottom_corner.x = round(p0[p0.size() - 1].x);

            sort(p0.begin(), p0.end(), [](cv::Point2f const& a, cv::Point2f const& b) { return a.y < b.y; });

            p_left_top_corner.y = round(p0[0].y);
            p_right_bottom_corner.y = round(p0[p0.size() - 1].y);

            cv::Point2i p_center_of_obj;
            p_center_of_obj.x = (p_left_top_corner.x + p_right_bottom_corner.x) / 2;
            p_center_of_obj.y = (p_left_top_corner.y + p_right_bottom_corner.y) / 2;

            // поиск объекта в текущем кадре
            cv::Point2i left_top_corner, right_bottom_corner;
            sort(p1.begin(), p1.end(), [](cv::Point2f const& a, cv::Point2f const& b) { return a.x < b.x; });

            left_top_corner.x = round(p1[0].x);
            right_bottom_corner.x = round(p1[p1.size() - 1].x);

            sort(p1.begin(), p1.end(), [](cv::Point2f const& a, cv::Point2f const& b) { return a.y < b.y; });

            left_top_corner.y = round(p1[0].y);
            right_bottom_corner.y = round(p1[p1.size() - 1].y);


            // поиск границ объекта и границ области интереса (ROI) для запуска нейронной сети
            if (abs(left_top_corner.x - right_bottom_corner.x) < currentFrame.size().width && abs(left_top_corner.y - right_bottom_corner.y) < currentFrame.size().height)
            {
                
                cv::Point2f center_of_obj;
                center_of_obj.x = (left_top_corner.x + right_bottom_corner.x) / 2;
                center_of_obj.y = (left_top_corner.y + right_bottom_corner.y) / 2;

                prevCoordinates.push_back(center_of_obj);

                int r_x = std::max(left_top_corner.x - 100, 1);
                int r_y = std::max(left_top_corner.y - 100, 1);
                int r_sx = std::min(abs(right_bottom_corner.x - left_top_corner.x) + 200, currentFrame.size().width - r_x - 1);
                int r_sy = std::min(abs(right_bottom_corner.y - left_top_corner.y) + 200, currentFrame.size().height - r_y - 1);

                cv::Rect roi = cv::Rect(r_x, r_y, r_sx, r_sy);

                std::cout << roi << "\n";

                // запуск распознавания объектов в области интереса с помощью нейронной сети
                resultPoints = dnn_execution(currentFrame, roi);

                cv::Scalar color = {0,0,0};

                for (int i = 0; i<resultPoints.size(); i++)
                {
                    cv::String caption = "";
                    switch (resultPoints[i].tType)
                    {
                    case TargetType::UAVTypePlane:
                        caption = class_names[2];//"Plane";
                        color = colors[0];
                        break;
                    case TargetType::Bird:
                        caption = class_names[0]; //"Bird";
                        color = colors[1];
                        break;
                    case TargetType::UAVTypeQuadcopter:
                        caption = class_names[1];//"Quadcopter";
                        color = colors[2];
                        break;
                    default:
                        caption = class_names[3];//"Undefined";
                        color = colors[3];
                        break;
                    }
                    cv::rectangle(currentFrame, cv::Point(resultPoints[i].x, resultPoints[i].y),
                        cv::Point(resultPoints[i].x+ resultPoints[i].sizeX, resultPoints[i].y + resultPoints[i].sizeY), 
                        color, 3);
                    cv::putText(currentFrame, caption, cv::Point(resultPoints[i].x, resultPoints[i].y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                }

                //rectangle(currentFrame, left_top_corner, right_bottom_corner, CV_RGB(255, 0, 0), 2, 8, 0);

                //resultPoints.push_back(curResult);

                //line(currentFrame, center_of_obj, p_center_of_obj, CV_RGB(0, 255, 255), 2);
            }
            else
                movPoints = 0;
            
            // пока что экстраполяция отображается прямо на кадре видео, так как в ТЗ не был прописан механизм передачи предсказываемых координат
            //!!! или это должно добавляться в качестве поправки к самим координатам объекта?
            if (prevCoordinates.size() > prevFrameCount)
            {
                cv::Point2f pred_point = line_interpolation(prevCoordinates, prevFrameCount, extraFrameCount);

                line(currentFrame, prevCoordinates[prevCoordinates.size() - 1], pred_point, CV_RGB(0, 0, 255), 2);
                circle(currentFrame, pred_point, 5, CV_RGB(0, 0, 255), 2);

            }

            p0 = good_new;

            // поиск движущихся объектов в кадре перезапускается, если все движущиеся объекты будут потеряны 
            //!!! можно усложнить условие, но пока нет четкого понимания, какие условия более корректны, еще думаю над этим
            if (movPoints < 1)
            {
                LastFrame = currentFrame.clone();
                cv::cvtColor(LastFrame, old_gray, cv::COLOR_BGR2GRAY);
                goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, cv::Mat(), 3, false, 0.04); //!!! возможно, часть этих параметров вынести в конструктор?
                prevCoordinates.clear();
            }
        }

        LastFrame = currentFrame.clone();
        return resultPoints;
    }
};
