/**
 *  @file test_alignment.cpp
 *  @brief Registration between depth and color image.
 *  @author haofan ren, yqykrhf@163.com
 *  @version beta 0.0
 *  @date 22-6-7
 *  @todo ReadSceneTxtFile() function
 */

// std
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>
using std::cout;
using std::endl;
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
// file operation
#include <dirent.h>
#ifdef _WIN32
#include <io.h>
#elif __linux__
#include <inttypes.h>
#include <unistd.h>
#define __int64 int64_t
#define _close close
#define _read read
#define _lseek64 lseek64
#define _O_RDONLY O_RDONLY
#define _open open
#define _lseeki64 lseek64
#define _lseek lseek
#define stricmp strcasecmp
#endif
#include <fstream>

#define USING_FLOAT64

// Flag using float or double as basic scalar data type.
#ifdef USING_FLOAT64
typedef double scalar;
typedef cv::Matx44d Mat4x4;
typedef cv::Matx14d Mat14;
#else
typedef float scalar;
typedef cv::Matx44f Mat4x4;
typedef cv::Matx14f Mat14;
#endif

/**
 * @brief Print OpenCV version.
 */
void PrintOpenCVInfo(void);
/**
 * @brief Print OpenCV Mat data type.
 * @param src Mat to extract data type.
 * @param s Prompt information indicate which matrix.
 */
void PrintMatType(const cv::Mat& src, const std::string& s = "Matrix");

/**
 * @brief Store the sensor intrinsic matrix and dpeth scale.
 */
struct CameraParameters {
#ifdef USING_FLOAT64
  Mat4x4 color_intrinsic4x4 = cv::Matx44d::eye();
  Mat4x4 depth_intrinsic4x4 = cv::Matx44d::eye();
  Mat4x4 d2c_extrinsic4x4 = cv::Matx44d::eye();
#else
  Mat4x4 color_intrinsic4x4 = cv::Matx44f::eye();
  Mat4x4 depth_intrinsic4x4 = cv::Matx44f::eye();
  Mat4x4 d2c_extrinsic4x4 = cv::Matx44f::eye();
#endif
  scalar depth_scale;
  const int cout_precision = 6;
  const int cout_w = 12;

  CameraParameters(void) {}
  ~CameraParameters(void) {}

  void SetIntrinsics(const scalar& cfx, const scalar& cfy, const scalar& cmx,
                     const scalar& cmy, const scalar& dfx, const scalar& dfy,
                     const scalar& dmx, const scalar& dmy,
                     const scalar& scale = 1000.0) {
    // initialize color instrinstc matrix.
    color_intrinsic4x4(0, 0) = cfx;
    color_intrinsic4x4(1, 1) = cfy;
    color_intrinsic4x4(0, 2) = cmx;
    color_intrinsic4x4(1, 2) = cmy;

    // initialize depth instrinstc matrix.
    depth_intrinsic4x4(0, 0) = dfx;
    depth_intrinsic4x4(1, 1) = dfy;
    depth_intrinsic4x4(0, 2) = dmx;
    depth_intrinsic4x4(1, 2) = dmy;
    depth_scale = scale;
  }

#ifdef USING_FLOAT64
  void SetExtrinsic(const Mat4x4& mat) { d2c_extrinsic4x4 = mat; }
#else
  void SetExtrinsic(const Mat4x4& mat) { d2c_extrinsic4x4 = mat; }
#endif

  scalar GetColorFx(void) { return color_intrinsic4x4(0, 0); }
  scalar GetColorFy(void) { return color_intrinsic4x4(1, 1); }
  scalar GetColorMx(void) { return color_intrinsic4x4(0, 2); }
  scalar GetColorMy(void) { return color_intrinsic4x4(1, 2); }

  scalar GetDepthFx(void) { return depth_intrinsic4x4(0, 0); }
  scalar GetDepthFy(void) { return depth_intrinsic4x4(1, 1); }
  scalar GetDepthMx(void) { return depth_intrinsic4x4(0, 2); }
  scalar GetDepthMy(void) { return depth_intrinsic4x4(1, 2); }

  void PrintColorIntrinsic(void) {
    cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    for (int i = 0; i < 4; ++i) {
      cout << "[";
      for (int j = 0; j < 4; ++j) {
        cout << std::setw(cout_w) << std::setprecision(cout_precision)
             << color_intrinsic4x4(i, j);
        cout << " ";
      }
      cout << "]" << endl;
    }
    cout.unsetf(std::ios_base::fixed);
    cout.unsetf(std::ios_base::floatfield);
  }

  void PrintDepthIntrinsic(void) {
    cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    for (int i = 0; i < 4; ++i) {
      cout << "[";
      for (int j = 0; j < 4; ++j) {
        cout << std::setw(cout_w) << std::setprecision(cout_precision)
             << depth_intrinsic4x4(i, j);
        cout << " ";
      }
      cout << "]" << endl;
    }
    cout.unsetf(std::ios_base::fixed);
    cout.unsetf(std::ios_base::floatfield);
  }

  void PrintDepth2ColorExtrinsic(void) {
    cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    for (int i = 0; i < 4; ++i) {
      cout << "[";
      for (int j = 0; j < 4; ++j) {
        cout << std::setw(cout_w) << std::setprecision(cout_precision)
             << d2c_extrinsic4x4(i, j);
        cout << " ";
      }
      cout << "]" << endl;
    }
    cout.unsetf(std::ios_base::fixed);
    cout.unsetf(std::ios_base::floatfield);
  }
};

/**
 * @brief ComposeOmegaMatrix
 * @param sensor
 * @param omega
 */
void ComposeOmegaMatrix(const CameraParameters& sensor, Mat4x4& omega);

/**
 * @brief ReadSceneTxtFile
 * @param path Scene txt file path, i.e. scene0000_00.txt
 * @param sensor Store the paramters extracted from txt file.
 */
void ReadSceneTxtFile(const std::string& path, CameraParameters& sensor);

int main(int argc, char** argv) {
  PrintOpenCVInfo();
  CameraParameters cp;
  std::string path = "/home/steve/dataset/scene0000_00/scene0000_00.txt";
  ReadSceneTxtFile(path, cp);
  cp.PrintColorIntrinsic();
  cout << "-------------------------------------------------------" << endl;
  cp.PrintDepthIntrinsic();
  cout << "-------------------------------------------------------" << endl;
  cp.PrintDepth2ColorExtrinsic();
  cout << "-------------------------------------------------------" << endl;

  Mat4x4 omega;
  ComposeOmegaMatrix(cp, omega);
  cout << omega << endl;
  cout << "-------------------------------------------------------" << endl;

  std::string color_path = "/home/steve/dataset/scene0000_00/color/0.jpg";
  std::string depth_path = "/home/steve/dataset/scene0000_00/depth/0.png";

  cv::Mat color_img = cv::imread(color_path, -1);
  cv::Mat depth_img = cv::imread(depth_path, -1);
  if (color_img.empty() || depth_img.empty()) {
    fprintf(stderr, "Error");
    exit(1);
  }
  PrintMatType(color_img, "color image data type");
  PrintMatType(depth_img, "depth image data type");

  cv::Mat aligned_color(depth_img.size(), color_img.type(),
                        cv::Scalar(0, 0, 0));

  cv::Mat color_img_resize;
  cv::resize(color_img, color_img_resize, depth_img.size(), cv::INTER_LINEAR);

  //  std::vector<cv::Vec2i> index;
  for (int v = 0; v < depth_img.rows; ++v) {
    // depth image
    unsigned short* drow_ptr = depth_img.ptr<unsigned short>(v);
    // align color image
    unsigned char* acrow_ptr = aligned_color.ptr<unsigned char>(v);
    cv::Vec2i idx;
    for (int u = 0; u < depth_img.cols; ++u) {
      // compute correpondent align color index from depth image index
      unsigned short depth = drow_ptr[u];
      if (depth == 0) continue;
      // real dpeth is depth / scale, (for example 1000.0)
      // r_depth: reciprocal of real depth
      scalar r_depth = 1000.0 / static_cast<scalar>(depth);
      scalar tmpu = omega.row(0).dot(Mat14(u, v, 1, r_depth));
      scalar tmpv = omega.row(1).dot(Mat14(u, v, 1, r_depth));

      idx[0] = static_cast<int>(std::round(tmpu));
      idx[1] = static_cast<int>(std::round(tmpv));
      // check corner case
      idx[0] = (idx[0] > (depth_img.cols - 1)) ? (depth_img.cols - 1) : idx[0];
      idx[1] = (idx[1] > (depth_img.rows - 1)) ? (depth_img.rows - 1) : idx[1];
      if (idx[0] > (depth_img.cols - 1)) {
        cout << "ERROR" << idx[0];
      }
      if (idx[1] > (depth_img.rows - 1)) {
        cout << "ERROR" << idx[1];
      }

      //      cout << idx[1] << " " << idx[0] << endl;
      //      index.push_back(idx);

      unsigned char* acdata_ptr = &acrow_ptr[u * aligned_color.channels()];
      //      acdata_ptr[0] = 255;
      //      acdata_ptr[1] = 0;
      //      acdata_ptr[2] = 0;
      acdata_ptr[0] = color_img.at<cv::Vec3b>(idx[1], idx[0])[0];
      acdata_ptr[1] = color_img.at<cv::Vec3b>(idx[1], idx[0])[1];
      acdata_ptr[2] = color_img.at<cv::Vec3b>(idx[1], idx[0])[2];
    }
  }

  cv::imshow("aligne color", aligned_color);
  cv::waitKey(0);
  cv::imwrite("test.png", aligned_color);
  return 0;
}

void PrintOpenCVInfo(void) {
  cout << "OpenCV Version:";
  cout << CV_VERSION << endl;
  cout << "------------------" << endl;
}
void PrintMatType(const cv::Mat& src, const std::string& s) {
  /* +-------- + ---- + ---- + ---- + ---- + ------ + ------ + ------ + ------ +
  |        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
  +-------- + ---- + ---- + ---- + ---- + ------ + ------ + ------ + ------ +
  | CV_8U | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 |
  | CV_8S | 1 | 9 | 17 | 25 | 33 | 41 | 49 | 57 |
  | CV_16U | 2 | 10 | 18 | 26 | 34 | 42 | 50 | 58 |
  | CV_16S | 3 | 11 | 19 | 27 | 35 | 43 | 51 | 59 |
  | CV_32S | 4 | 12 | 20 | 28 | 36 | 44 | 52 | 60 |
  | CV_32F | 5 | 13 | 21 | 29 | 37 | 45 | 53 | 61 |
  | CV_64F | 6 | 14 | 22 | 30 | 38 | 46 | 54 | 62 |
  +-------- + ---- + ---- + ---- + ---- + ------ + ------ + ------ + ------ +*/
  if (src.empty()) {
    cout << "PrintMatType: image is empty";
    return;
  }
  int type = src.type();
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  r += "C";
  r += (chans + '0');
  printf("%7s: %7s, (col,row):%dx%d \n", s.c_str(), r.c_str(), src.cols,
         src.rows);
}

void ReadSceneTxtFile(const std::string& path, CameraParameters& sensor) {
  if (path.empty()) {
    fprintf(stderr, "File %s, Line %d, Funcion %s(), path: %s is empty!\n",
            __FILE__, __LINE__, __FUNCTION__, path.c_str());
    exit(1);
  }
  std::ifstream f(path);
  if (!f.is_open()) {
    fprintf(stderr, "File %s, Line %d, Funcion %s(), Failed to open file: %s\n",
            __FILE__, __LINE__, __FUNCTION__, path.c_str());
    exit(1);
  }
  //  std::string s;
  //  f >> s;
  sensor.SetIntrinsics(1170.187988, 1170.187988, 647.750000, 483.750000,
                       571.623718, 571.623718, 319.500000, 239.500000);
  Mat4x4 m(0.999973, 0.006791, 0.002776, -0.037886, -0.006767, 0.999942,
           -0.008366, -0.003410, -0.002833, 0.008347, 0.999961, -0.021924,
           -0.000000, 0.000000, -0.000000, 1.000000);

  sensor.SetExtrinsic(m.inv());
  f.close();
}

void ComposeOmegaMatrix(const CameraParameters& sensor, Mat4x4& mat) {
  mat = sensor.color_intrinsic4x4 * sensor.d2c_extrinsic4x4 *
        sensor.depth_intrinsic4x4.inv();
}
