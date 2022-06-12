`Data_loader.cpp`

```c++
void Data_loader::load_scannet_ground_truth()
vector<string>getFiles_scannet()
```

`Image_loader.cpp`直接resize, toalign color and depth image

```C++
case ImageLoaderMode::WITH_COLOR_AND_DEPTH:
        {
            depth_mat = cv::imread(this->depth_path_vector[this->frame_index].c_str(), CV_LOAD_IMAGE_UNCHANGED);

            std::cout<<"该深度圖："<<this->depth_path_vector[this->frame_index].c_str()<<std::endl;

            // cv::pyrUp(depth_mat1, depth_mat, cvSize(depth_mat1.cols * 2, depth_mat1.rows * 2));
            std::cout<<"深度圖尺寸："<<depth_mat.cols<<std::endl;
            std::cout<<"深度圖尺寸："<<depth_mat.rows<<std::endl;
            cv::Mat temp = cv::imread(this->color_path_vector[this->frame_index].c_str(), CV_LOAD_IMAGE_UNCHANGED);
            cv::resize(temp, color_mat1, cvSize(640,480));
//            pyrDown(temp, color_mat1, cvSize(temp.cols / 4.05, temp.rows / 4.05));
            //cv::pyrUp(temp, color_mat1, cvSize(temp.cols * 2, temp.rows * 2));
            cv::cvtColor(color_mat1, color_mat, cv::COLOR_BGRA2BGR);
            std::cout<<"彩色圖尺寸："<<color_mat.cols<<std::endl;
            std::cout<<"彩色圖尺寸："<<color_mat.rows<<std::endl;
//            cv::imshow("color_mat:",color_mat);
//            cv::waitKey(0);
            break;
        }
```

`Map_engine`

加入了三个成员变量

```c++
    My_Type::Vector3f * scene_points_offscreen;
    My_Type::Vector4uc  * object_points_color;
    My_Type::Vector3f * scene_points_offscreen;
```

记得析构函数释放掉











