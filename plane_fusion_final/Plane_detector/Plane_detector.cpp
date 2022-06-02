#include "Plane_detector.h"

//
#include "SLAM_system/SLAM_system_settings.h"
//
#include "Plane_detector_KernelFunc.cuh"

//
#include <float.h>

#include <cstdio>
#include <cstring>
#include <iostream>

#include "math.h"

// ---------------------------- Plane_detector (base class)
#pragma region(Plane_detector base class)

//
Plane_detector::Plane_detector() { this->current_plane_counter = 0; }
//
Plane_detector::~Plane_detector() {
  // Release HOST memory
  free(this->cell_info_mat);
  free(this->current_planes);
  free(this->model_planes);
  free(this->relative_matrix);

  // Release CUDA memory
  checkCudaErrors(cudaFree(this->dev_cell_info_mat));
  checkCudaErrors(cudaFree(this->dev_current_cell_labels));
  checkCudaErrors(cudaFree(this->dev_current_plane_labels));
  checkCudaErrors(cudaFree(this->dev_current_planes));
  checkCudaErrors(cudaFree(this->dev_model_planes));
  checkCudaErrors(cudaFree(this->dev_current_plane_counter));
  checkCudaErrors(cudaFree(this->dev_relative_matrix));
  checkCudaErrors(cudaFree(this->dev_matches));
  checkCudaErrors(cudaFree(this->dev_buffer_coordinate));
}

//
void Plane_detector::init() {
  this->aligned_depth_size =
      SLAM_system_settings::instance()->aligned_depth_size;
  this->cell_mat_size = this->aligned_depth_size /
                        SLAM_system_settings::instance()->presegment_cell_width;

  //
  this->cell_info_mat = (Cell_info *)malloc(
      this->cell_mat_size.x * this->cell_mat_size.y * sizeof(Cell_info));
  this->current_planes =
      (Plane_info *)malloc(MAX_CURRENT_PLANES * sizeof(Plane_info));
  this->model_planes =
      (Plane_info *)malloc(MAX_MODEL_PLANES * sizeof(Plane_info));
  this->relative_matrix =
      (int *)malloc(MAX_CURRENT_PLANES * MAX_MODEL_PLANES * sizeof(int));

  //
  checkCudaErrors(cudaMalloc(
      (void **)&(this->dev_cell_info_mat),
      this->cell_mat_size.x * this->cell_mat_size.y * sizeof(Cell_info)));
  checkCudaErrors(cudaMalloc(
      (void **)&(this->dev_current_cell_labels),
      this->aligned_depth_size.x * this->aligned_depth_size.y * sizeof(int)));
  checkCudaErrors(cudaMalloc(
      (void **)&(this->dev_current_plane_labels),
      this->aligned_depth_size.x * this->aligned_depth_size.y * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_current_planes),
                             MAX_CURRENT_PLANES * sizeof(Plane_info)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_model_planes),
                             MAX_MODEL_PLANES * sizeof(Plane_info)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_current_plane_counter), sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_relative_matrix),
                 MAX_CURRENT_PLANES * MAX_MODEL_PLANES * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_matches),
                             MAX_CURRENT_PLANES * sizeof(My_Type::Vector2i)));

  // Buffer coordinate
  checkCudaErrors(cudaMalloc((void **)&(this->dev_buffer_coordinate),
                             MAX_CURRENT_PLANES * sizeof(Plane_coordinate)));

  //
  sdkCreateTimer(&this->timer_average);
  sdkResetTimer(&this->timer_average);
}

//
void Plane_detector::detect_plane(const My_Type::Vector3f *dev_current_points,
                                  const My_Type::Vector3f *dev_current_normals,
                                  const Eigen::Matrix4f &camera_pose,
                                  const Plane_info *dev_model_planes,
                                  int *dev_previous_plane_labels,
                                  bool with_continuous_frame_tracking) {
  static bool first_reach = true;
  if (first_reach) {
    first_reach = false;
    sdkResetTimer(&this->timer_average);
  }
  sdkStartTimer(&(this->timer_average));

  // 0. Prepare : clean memory
  this->prepare_to_detect();

  // 1. Pre-segmentation
  this->presegment_to_cell(dev_current_points, dev_current_normals);

  // 2. Fit plane for each cell
  this->fit_plane_for_each_cell(dev_current_points, dev_current_normals,
                                this->dev_cell_info_mat);

  // 3. Cluster cells to detect plane
  this->cluster_cells(this->dev_cell_info_mat, dev_model_planes,
                      this->dev_current_planes, with_continuous_frame_tracking);

  sdkStopTimer(&(this->timer_average));
  float elapsed_time = sdkGetAverageTimerValue(&(this->timer_average));
  // printf("elapsed_time = %f\n", elapsed_time);
  // printf("%f\n", elapsed_time);

  //
  // 4. Transfer current plane coordiante to map
  this->transfer_plane_coordinate(camera_pose);
}

//
void Plane_detector::transfer_plane_coordinate(
    const Eigen::Matrix4f &camera_pose) {
  for (int plane_index = 1; plane_index < this->current_plane_counter;
       plane_index++) {
    Eigen::Vector3f current_normal_C;
    float current_d_C;

    //
    current_normal_C.x() = this->current_planes[plane_index].nx;
    current_normal_C.y() = this->current_planes[plane_index].ny;
    current_normal_C.z() = this->current_planes[plane_index].nz;
    current_d_C = this->current_planes[plane_index].d;
    //
    Eigen::Vector3f current_normal_W;
    float current_d_W;
    current_normal_W = camera_pose.block(0, 0, 3, 3) * current_normal_C;
    //
    Eigen::Vector3f current_plane_W;
    current_plane_W =
        -current_normal_W * current_d_C + camera_pose.block(0, 3, 3, 1);
    current_d_W = -current_plane_W.dot(current_normal_W);

    //
    this->current_planes[plane_index].nx = current_normal_W.x();
    this->current_planes[plane_index].ny = current_normal_W.y();
    this->current_planes[plane_index].nz = current_normal_W.z();
    this->current_planes[plane_index].d = current_d_W;
  }

  //! Debug
  if (1) {
    //
    for (int plane_id = 1; plane_id < this->current_plane_counter; plane_id++) {
      printf(
          "%d : (%f, %f, %f), %f , %f , %d\t", plane_id,
          this->current_planes[plane_id].nx, this->current_planes[plane_id].ny,
          this->current_planes[plane_id].nz, this->current_planes[plane_id].d,
          this->current_planes[plane_id].area,
          this->current_planes[plane_id].cell_num);

      if (this->current_planes[plane_id].is_valid) {
        printf("true \n");
      } else {
        printf("false \n");
      }
    }
    printf("\n");
  }
}

//
void Plane_detector::match_planes(const Plane_info *model_planes,
                                  int model_plane_number,
                                  const int *dev_current_plane_labels,
                                  const int *dev_model_plane_labels) {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  if (this->current_plane_counter == 0 ||
      this->current_plane_counter >= MAX_CURRENT_PLANES)
    return;

  // 0. init data
  {
    for (int plane_id = 0; plane_id < this->current_plane_counter; plane_id++) {
      this->current_planes[plane_id].pixel_num = 0;
    }

    for (int plane_id = 0; plane_id < model_plane_number; plane_id++) {
      this->model_planes[plane_id] = model_planes[plane_id];
      this->model_planes[plane_id].pixel_num = 0;
    }

    checkCudaErrors(cudaMemcpy(this->dev_current_planes, this->current_planes,
                               MAX_CURRENT_PLANES * sizeof(Plane_info),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->dev_model_planes, this->model_planes,
                               MAX_MODEL_PLANES * sizeof(Plane_info),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(
        cudaMemset(this->dev_relative_matrix, 0x00,
                   MAX_CURRENT_PLANES * MAX_MODEL_PLANES * sizeof(int)));
  }

  // 1. Count pixel number of each plane
  {
    thread_rect.x = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.y = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.z = 1;
    block_rect.x = this->cell_mat_size.width;
    block_rect.y = this->cell_mat_size.height;
    block_rect.z = 1;

    // Lunch kernel function
    count_planar_pixel_number_CUDA(
        block_rect, thread_rect, dev_current_plane_labels,
        this->dev_current_planes, this->current_plane_counter);
    // CUDA_CKECK_KERNEL;

    // Lunch kernel function
    count_planar_pixel_number_CUDA(block_rect, thread_rect,
                                   dev_model_plane_labels,
                                   this->dev_model_planes, model_plane_number);
    // CUDA_CKECK_KERNEL;

    checkCudaErrors(cudaMemcpy(this->current_planes, this->dev_current_planes,
                               MAX_CURRENT_PLANES * sizeof(Plane_info),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(this->model_planes, this->dev_model_planes,
                               MAX_MODEL_PLANES * sizeof(Plane_info),
                               cudaMemcpyDeviceToHost));
    // for (int i = 0; i < this->current_plane_counter; i++)
    //	printf("%d, ", this->current_planes[i].pixel_number);
    // printf("\n");
    // for (int i = 0; i < model_plane_number; i++)
    //	printf("%d, ", this->model_planes[i].pixel_number);
    // printf("\n");
  }

  // 2. Count overlap pixel number
  {
    thread_rect.x = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.y = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.z = 1;
    block_rect.x = this->cell_mat_size.width;
    block_rect.y = this->cell_mat_size.height;
    block_rect.z = 1;

    // Lunch kernel function
    count_overlap_pixel_number_CUDA(
        block_rect, thread_rect, dev_current_plane_labels,
        dev_model_plane_labels, this->current_plane_counter,
        this->dev_relative_matrix);
    // CUDA_CKECK_KERNEL;

    // Copy out relative matrix
    checkCudaErrors(
        cudaMemcpy(this->relative_matrix, this->dev_relative_matrix,
                   MAX_CURRENT_PLANES * MAX_MODEL_PLANES * sizeof(int),
                   cudaMemcpyDeviceToHost));
  }

  // 3. Match current planes to model planes
  {
    this->matches.clear();
    // this->matches.reserve(this->current_plane_counter);
    this->matches.push_back(My_Type::Vector2i(0, 0));

    // for (int current_plane_id = 0; current_plane_id <
    // this->current_plane_counter; current_plane_id++)
    //{
    //	//Plane_info current_plane_info =
    // this->current_planes[current_plane_id]; 	for (int model_plane_id = 0;
    // model_plane_id < std::min(model_plane_number, MAX_MODEL_PLANES);
    // model_plane_id++)
    //	{
    //		//Plane_info model_plane_info =
    // this->model_planes[model_plane_id]; 		printf("%d | ",
    // this->relative_matrix[current_plane_id * MAX_MODEL_PLANES +
    // model_plane_id]);
    //	}
    //	printf("\n");
    //}

    //
    int new_model_plane_id = model_plane_number;
    for (int current_plane_id = 1;
         current_plane_id < this->current_plane_counter; current_plane_id++) {
      //
      bool is_valid_plane = this->current_planes[current_plane_id].is_valid;
      if (!is_valid_plane) {
        //
        My_Type::Vector2i plane_id_pair(current_plane_id, 0);
        this->matches.push_back(plane_id_pair);
      } else {
        bool is_new_plane = false;
        bool is_valid_match = true;

        std::vector<std::pair<int, int>> overlap_list =
            find_most_overlap_model_plane(current_plane_id, model_plane_number);
        int model_plane_id;
        int overlap_pixel_number = 0;

        if (overlap_list.size() == 1) {
          model_plane_id = overlap_list.front().first;
          overlap_pixel_number = overlap_list.front().second;
          if (model_plane_id == 0) is_new_plane = true;
        } else if (overlap_list.size() > 1) {
          if (overlap_list.front().first == 0 &&
              overlap_list[1].second > MIN_CELLS_OF_PLANE *
                                           PLANE_PIXEL_BLOCK_SIZE *
                                           TRACK_PLANE_THRESHOLD) {
            // Tracked plane
            model_plane_id = overlap_list[1].first;
            overlap_pixel_number = overlap_list[1].second;
          } else if (overlap_list.front().first != 0 &&
                     overlap_list.front().second > MIN_CELLS_OF_PLANE *
                                                       PLANE_PIXEL_BLOCK_SIZE *
                                                       TRACK_PLANE_THRESHOLD) {
            // Tracked plane
            model_plane_id = overlap_list.front().first;
            overlap_pixel_number = overlap_list.front().second;
          } else if (overlap_list.front().first == 0 &&
                     overlap_list.front().second >
                         MIN_CELLS_OF_PLANE * PLANE_PIXEL_BLOCK_SIZE) {
            // New plane
            model_plane_id = overlap_list.front().first;
            overlap_pixel_number = overlap_list.front().second;
            is_new_plane = true;
          } else {
            is_valid_match = false;
          }
        } else {
          is_valid_match = false;
        }

        if (is_new_plane && is_valid_match) {
          //
          My_Type::Vector2i plane_id_pair(current_plane_id, new_model_plane_id);
          new_model_plane_id++;
          this->matches.push_back(plane_id_pair);
          // printf("new : ");
        } else if (is_valid_match) {
          My_Type::Vector3f model_normal(model_planes[model_plane_id].nx,
                                         model_planes[model_plane_id].ny,
                                         model_planes[model_plane_id].nz);
          My_Type::Vector3f current_normal(
              this->current_planes[current_plane_id].nx,
              this->current_planes[current_plane_id].ny,
              this->current_planes[current_plane_id].nz);
          if (model_normal.dot(current_normal) < 0.9) is_valid_match = false;
          if (fabsf(model_planes[model_plane_id].d -
                    this->current_planes[current_plane_id].d) > 0.1)
            is_valid_match = false;

          if (is_valid_match) {
            My_Type::Vector2i plane_id_pair(current_plane_id, model_plane_id);
            this->matches.push_back(plane_id_pair);
          } else {
            // Invalid plane
            My_Type::Vector2i plane_id_pair(current_plane_id, 0);
            this->matches.push_back(plane_id_pair);
            this->current_planes[current_plane_id].is_valid = false;
          }

        } else {
          // Invalid plane
          My_Type::Vector2i plane_id_pair(current_plane_id, 0);
          this->matches.push_back(plane_id_pair);
          this->current_planes[current_plane_id].is_valid = false;
        }
      }
      // printf("%d : %d, %d\n", current_plane_id,
      // this->matches[current_plane_id].x, this->matches[current_plane_id].y);
    }
  }

  // 4. Re-label current plane labels to model plane order
  {
    checkCudaErrors(
        cudaMemcpy(this->dev_matches, this->matches.data(),
                   this->current_plane_counter * sizeof(My_Type::Vector2i),
                   cudaMemcpyHostToDevice));

    thread_rect.x = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.y = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.z = 1;
    block_rect.x = this->cell_mat_size.width;
    block_rect.y = this->cell_mat_size.height;
    block_rect.z = 1;
    relabel_plane_labels_CUDA(block_rect, thread_rect, this->dev_matches,
                              this->dev_current_plane_labels);
    // CUDA_CKECK_KERNEL;
  }
}

void Plane_detector::match_planes_to_new_map(
    const Plane_info *model_planes, int model_plane_number,
    const int *dev_current_plane_labels, const int *dev_model_plane_labels,
    std::vector<std::pair<int, int>> &plane_matches) {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  if (this->current_plane_counter == 0 ||
      this->current_plane_counter >= MAX_CURRENT_PLANES)
    return;

  // 0. init data
  {
    for (int plane_id = 0; plane_id < this->current_plane_counter; plane_id++) {
      this->current_planes[plane_id].pixel_num = 0;
    }

    for (int plane_id = 0; plane_id < model_plane_number; plane_id++) {
      this->model_planes[plane_id] = model_planes[plane_id];
      this->model_planes[plane_id].pixel_num = 0;
    }

    checkCudaErrors(cudaMemcpy(this->dev_current_planes, this->current_planes,
                               MAX_CURRENT_PLANES * sizeof(Plane_info),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(this->dev_model_planes, this->model_planes,
                               MAX_MODEL_PLANES * sizeof(Plane_info),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(
        cudaMemset(this->dev_relative_matrix, 0x00,
                   MAX_CURRENT_PLANES * MAX_MODEL_PLANES * sizeof(int)));
  }

  //
  plane_matches.clear();
  std::vector<int> valid_plane_index_list;
  valid_plane_index_list.push_back((int)0);
  for (int current_plane_id = 0, new_id = 1;
       current_plane_id < this->matches.size(); current_plane_id++) {
    printf("%d -> %d", this->matches[current_plane_id].x,
           this->matches[current_plane_id].y);
    if (this->matches[current_plane_id].y != 0) {
      plane_matches.push_back(
          std::pair<int, int>(this->matches[current_plane_id].y, new_id));
      valid_plane_index_list.push_back(current_plane_id);
      // printf("%d -> %d\n", this->matches[current_plane_id].y, new_id);
      printf(" --> %d", new_id);
      new_id++;
    }
    printf("\n");
  }

  // Relabel model_plane_id -> new_plane_id
  this->matches.resize(model_plane_number + 1);
  memset(this->matches.data(), 0x00,
         this->matches.size() * sizeof(My_Type::Vector2i));
  for (int i = 0; i < this->matches.size(); i++)
    this->matches[i] = My_Type::Vector2i(i, i);
  //
  for (int i = 0; i < plane_matches.size(); i++) {
    this->matches[plane_matches[i].first] = plane_matches[i].second;
  }
  // 4. Re-label current plane labels to model plane order
  {
    checkCudaErrors(cudaMemcpy(this->dev_matches, this->matches.data(),
                               this->matches.size() * sizeof(My_Type::Vector2i),
                               cudaMemcpyHostToDevice));

    thread_rect.x = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.y = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.z = 1;
    block_rect.x = this->cell_mat_size.width;
    block_rect.y = this->cell_mat_size.height;
    block_rect.z = 1;
    relabel_plane_labels_CUDA(block_rect, thread_rect, this->dev_matches,
                              this->dev_current_plane_labels);
    // CUDA_CKECK_KERNEL;
  }

  // 5. reverse match to new map
  this->matches.resize(valid_plane_index_list.size());
  memset(this->matches.data(), 0x00,
         this->matches.size() * sizeof(My_Type::Vector2i));
  for (int i = 0; i < this->matches.size(); i++) {
    this->matches[i] = My_Type::Vector2i(valid_plane_index_list[i], i);
    // this->matches[i] = My_Type::Vector2i(this->matches[i].x,
    // this->matches[i].y);
    printf("%d ==> %d\n", this->matches[i].x, this->matches[i].y);
  }

  //
}

//
int cmp_function(const void *a, const void *b) {
  if ((*(std::pair<int, int> *)a).second > (*(std::pair<int, int> *)b).second) {
    return -1;
  } else {
    return 1;
  }
}
//
std::vector<std::pair<int, int>> Plane_detector::find_most_overlap_model_plane(
    int current_plane_label, int model_plane_counter) {
  std::vector<std::pair<int, int>> overlap_list;
  //
  int overlap_pixel_number = 0;
  int overlap_plane_index = 0;
  //
  for (int model_index = 0; model_index < model_plane_counter; model_index++) {
    int tmep_pixel_number =
        this->relative_matrix[current_plane_label * MAX_MODEL_PLANES +
                              model_index];
    if (tmep_pixel_number > 0) {
      overlap_list.push_back(
          std::pair<int, int>(model_index, tmep_pixel_number));
    }
  }
  // Sort
  std::qsort(overlap_list.data(), overlap_list.size(),
             sizeof(std::pair<int, int>), cmp_function);

  return overlap_list;
}

//
void Plane_detector::prepare_to_detect() {
  //
  memset(this->cell_info_mat, 0x00,
         this->cell_mat_size.x * this->cell_mat_size.y * sizeof(Cell_info));
  memset(this->current_planes, 0x00, MAX_CURRENT_PLANES * sizeof(Plane_info));

  //
  checkCudaErrors(cudaMemset(
      this->dev_cell_info_mat, 0x00,
      this->cell_mat_size.x * this->cell_mat_size.y * sizeof(Cell_info)));
  checkCudaErrors(cudaMemset(
      this->dev_current_cell_labels, 0x00,
      this->aligned_depth_size.x * this->aligned_depth_size.y * sizeof(int)));
  checkCudaErrors(cudaMemset(
      this->dev_current_plane_labels, 0x00,
      this->aligned_depth_size.x * this->aligned_depth_size.y * sizeof(int)));
  checkCudaErrors(cudaMemset(this->dev_current_planes, 0x00,
                             (int)MAX_CURRENT_PLANES * sizeof(Plane_info)));
  checkCudaErrors(
      cudaMemset(this->dev_current_plane_counter, 0x00, sizeof(int)));
  checkCudaErrors(
      cudaMemset(this->dev_matches, 0x00,
                 (int)MAX_CURRENT_PLANES * sizeof(My_Type::Vector2i)));
}

#pragma endregion

// ---------------------------- Plane detector use stereoprojection
#pragma region(Plane detector use stereoprojection)
//
Plane_stereoprojection_detector::Plane_stereoprojection_detector() {
  printf("Plane_stereoprojection_detector\n");
}
Plane_stereoprojection_detector::~Plane_stereoprojection_detector() {
  // Release HOST data
  free(this->hist_mat);

  // Release CUDA data
  checkCudaErrors(cudaFree(this->dev_hist_mat));
  checkCudaErrors(cudaFree(this->dev_hist_normals));
  checkCudaErrors(cudaFree(this->dev_hist_normal_counter));
  checkCudaErrors(cudaFree(this->dev_prj_distance_hist));
  checkCudaErrors(cudaFree(this->dev_plane_mean_parameters));
  checkCudaErrors(cudaFree(this->dev_ATA_upper_buffer));
  checkCudaErrors(cudaFree(this->dev_ATb_buffer));
}

//
void Plane_stereoprojection_detector::init() {
  // Initialize base class members
  Plane_detector::init();

  //
  this->hist_mat =
      (float *)malloc(HISTOGRAM_WIDTH * HISTOGRAM_WIDTH * sizeof(float));

  //
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_hist_mat),
                 HISTOGRAM_WIDTH * HISTOGRAM_WIDTH * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_hist_normals),
                             MAX_HIST_NORMALS * sizeof(Hist_normal)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_hist_normal_counter), sizeof(int)));
  checkCudaErrors(cudaMalloc(
      (void **)&(this->dev_prj_distance_hist),
      ceil_by_stride((int)(MAX_VALID_DEPTH_M / MIN_PLANE_DISTANCE), 256) *
          sizeof(int)));
  // Buffers for K-means iteration
  checkCudaErrors(cudaMalloc((void **)&(this->dev_plane_mean_parameters),
                             MAX_CURRENT_PLANES * sizeof(Cell_info)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_ATA_upper_buffer),
                             3 * MAX_CURRENT_PLANES * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_ATb_buffer),
                             2 * MAX_CURRENT_PLANES * sizeof(float)));
}

//
void Plane_stereoprojection_detector::prepare_to_detect() {
  //
  Plane_detector::prepare_to_detect();

  //
  memset(this->hist_mat, 0x00,
         HISTOGRAM_WIDTH * HISTOGRAM_WIDTH * sizeof(float));

  //
  checkCudaErrors(
      cudaMemset(this->dev_hist_mat, 0x00,
                 HISTOGRAM_WIDTH * HISTOGRAM_WIDTH * sizeof(float)));
  checkCudaErrors(cudaMemset(this->dev_hist_normals, 0x00,
                             MAX_HIST_NORMALS * sizeof(Hist_normal)));
  checkCudaErrors(cudaMemset(this->dev_hist_normal_counter, 0x00, sizeof(int)));
}

//
void Plane_stereoprojection_detector::fit_plane_for_each_cell(
    const My_Type::Vector3f *dev_current_points,
    const My_Type::Vector3f *dev_current_normals,
    Cell_info *dev_cell_info_mat) {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  // 1. fit plane for each cells
  {
    thread_rect.x = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.y = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.z = 1;
    block_rect.x = this->cell_mat_size.width;
    block_rect.y = this->cell_mat_size.height;
    block_rect.z = 1;
    // Lunch kernel function
    fit_plane_for_cells_CUDA(
        block_rect, thread_rect, dev_current_points, dev_current_normals,
        SLAM_system_settings::instance()->sensor_params, dev_cell_info_mat);
    // CUDA_CKECK_KERNEL;
  }
}

//
void Plane_stereoprojection_detector::cluster_cells(
    Cell_info *dev_cell_info_mat, const Plane_info *dev_model_planes,
    Plane_info *dev_current_planes, bool with_continuous_frame_tracking) {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

#pragma region(Generate plane paramter seed)
  // 1. Histogram normal direction of cells on PxPy(use stereoproject [nx, ny,
  // nz] to [px, py])
  {
    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x =
        this->cell_mat_size.width * this->cell_mat_size.height / thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    histogram_PxPy_CUDA(block_rect, thread_rect, dev_cell_info_mat,
                        this->dev_hist_mat);
    // CUDA_CKECK_KERNEL;

    // Copy out histogram
    checkCudaErrors(
        cudaMemcpy(this->hist_mat, this->dev_hist_mat,
                   HISTOGRAM_WIDTH * HISTOGRAM_WIDTH * sizeof(float),
                   cudaMemcpyDeviceToHost));
  }

  // 2. Search peaks on PxPy as plane normal direction
  {
    // Detect plane direction from last frame
    checkCudaErrors(cudaMemset(this->dev_hist_mat, 0x00,
                               MAX_HIST_NORMALS * sizeof(Hist_normal)));
    checkCudaErrors(
        cudaMemset(this->dev_hist_normal_counter, 0x00, sizeof(int)));
    //
    thread_rect.x = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.y = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.z = 1;
    block_rect.x = HISTOGRAM_WIDTH / thread_rect.x;
    block_rect.y = HISTOGRAM_WIDTH / thread_rect.y;
    block_rect.z = 1;
    // Lunch kernel function
    find_PxPy_peaks_CUDA(block_rect, thread_rect, this->dev_hist_mat,
                         this->dev_hist_normals, this->dev_hist_normal_counter);
    // CUDA_CKECK_KERNEL;
    // Copy out peaks
    checkCudaErrors(cudaMemcpy(&this->hist_normal_counter,
                               this->dev_hist_normal_counter, sizeof(int),
                               cudaMemcpyDeviceToHost));
  }

  // 3. Find planes in each direction
  {
    int init_counter = 1;
    checkCudaErrors(cudaMemcpy(this->dev_current_plane_counter, &init_counter,
                               sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(this->dev_current_planes, 0x00,
                               MAX_CURRENT_PLANES * sizeof(Plane_info)));
    //
    Hist_normal *dev_hist_normal_ptr = this->dev_hist_normals;
    // printf("hist_normal_counter = %d\r\n", this->hist_normal_counter);
    for (int i = 0; i < this->hist_normal_counter; i++) {
      // Reset distance histogram
      checkCudaErrors(cudaMemset(
          this->dev_prj_distance_hist, 0x00,
          ceil_by_stride((int)(MAX_VALID_DEPTH_M / MIN_PLANE_DISTANCE), 256) *
              sizeof(float)));
      // Set CUDA grid size
      thread_rect.x = 256;
      thread_rect.y = 1;
      thread_rect.z = 1;
      block_rect.x =
          ceil_by_stride(this->cell_mat_size.width * this->cell_mat_size.height,
                         256) /
          thread_rect.x;
      block_rect.y = 1;
      block_rect.z = 1;
      // Lunch kernel function
      histogram_prj_dist_CUDA(block_rect, thread_rect, this->dev_cell_info_mat,
                              dev_hist_normal_ptr, this->dev_prj_distance_hist);
      // CUDA_CKECK_KERNEL;

      // Set CUDA grid size
      thread_rect.x = 256;
      thread_rect.y = 1;
      thread_rect.z = 1;
      block_rect.x =
          ceil_by_stride((int)(MAX_VALID_DEPTH_M / MIN_PLANE_DISTANCE), 256) /
          thread_rect.x;
      block_rect.y = 1;
      block_rect.z = 1;
      // Lunch kernel function
      find_prj_dist_peaks_CUDA(block_rect, thread_rect,
                               this->dev_prj_distance_hist, dev_hist_normal_ptr,
                               this->dev_current_plane_counter,
                               this->dev_current_planes);
      // CUDA_CKECK_KERNEL;

      // Next normal pointer
      dev_hist_normal_ptr++;
    }

    // Copy out current plane number
    checkCudaErrors(cudaMemcpy(&this->current_plane_counter,
                               this->dev_current_plane_counter, sizeof(int),
                               cudaMemcpyDeviceToHost));
  }
#pragma endregion

#pragma region(K - means iteration)
  // 4. K-means iteration
  {
    const int kmeans_iteration_times = 5;
    for (int i = 0; i < kmeans_iteration_times; i++) {
      // Prepare for next iteration
      checkCudaErrors(cudaMemset(this->dev_plane_mean_parameters, 0x00,
                                 MAX_CURRENT_PLANES * sizeof(Cell_info)));
      checkCudaErrors(cudaMemset(this->dev_ATA_upper_buffer, 0x00,
                                 3 * MAX_CURRENT_PLANES * sizeof(float)));
      checkCudaErrors(cudaMemset(this->dev_ATb_buffer, 0x00,
                                 2 * MAX_CURRENT_PLANES * sizeof(float)));

      // Set CUDA grid size
      thread_rect.x = 256;
      thread_rect.y = 1;
      thread_rect.z = 1;
      block_rect.x =
          ceil_by_stride(this->cell_mat_size.width * this->cell_mat_size.height,
                         thread_rect.x) /
          thread_rect.x;
      block_rect.y = 1;
      block_rect.z = 1;
      // Lunch kernel function
      K_mean_iterate_CUDA(
          block_rect, thread_rect, this->dev_current_planes,
          this->dev_cell_info_mat, this->dev_plane_mean_parameters,
          this->dev_buffer_coordinate, this->dev_ATA_upper_buffer,
          this->dev_ATb_buffer, this->current_plane_counter);
      // CUDA_CKECK_KERNEL;
    }
  }
#pragma endregion

  // 5. Label current plane
  {
    //
    thread_rect.x = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.y = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.z = 1;
    block_rect.x = this->cell_mat_size.width;
    block_rect.y = this->cell_mat_size.height;
    block_rect.z = 1;
    //
    label_current_planes_CUDA(block_rect, thread_rect, this->dev_cell_info_mat,
                              this->dev_current_plane_labels);
    // CUDA_CKECK_KERNEL;
  }

  // Copy out current plane information
  checkCudaErrors(cudaMemcpy(&this->current_plane_counter,
                             this->dev_current_plane_counter, sizeof(int),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(this->current_planes, this->dev_current_planes,
                             this->current_plane_counter * sizeof(Plane_info),
                             cudaMemcpyDeviceToHost));

  // For render
  {
    checkCudaErrors(
        cudaMemcpy(this->cell_info_mat, this->dev_cell_info_mat,
                   this->cell_mat_size.width * this->cell_mat_size.height *
                       sizeof(Cell_info),
                   cudaMemcpyDeviceToHost));
  }
}

#pragma endregion

// ---------------------------- Plane detector use super pixel
Plane_super_pixel_detector::Plane_super_pixel_detector() {}
//
Plane_super_pixel_detector::~Plane_super_pixel_detector() {
  //
  free(this->super_pixel_adjacent_mat);

  // Release CUDA data
  checkCudaErrors(cudaFree(this->dev_super_pixel_id_image));
  checkCudaErrors(cudaFree(this->dev_super_pixel_mat));
  checkCudaErrors(cudaFree(this->dev_super_pixel_accumulate_mat));
  checkCudaErrors(cudaFree(this->dev_cell_hessain_uppers));
  checkCudaErrors(cudaFree(this->dev_cell_nabla));
  checkCudaErrors(cudaFree(this->dev_base_vectors));
}

//
void Plane_super_pixel_detector::init() {
  // Initialize base class members
  Plane_detector::init();

  //
  this->super_pixel_mat_size =
      this->aligned_depth_size /
      SLAM_system_settings::instance()->presegment_cell_width;

  //
  this->number_of_CUDA_block_per_line =
      ceilf((float)SLAM_system_settings::instance()->presegment_cell_width * 3 /
                (float)SUPER_PIXEL_BLOCK_WIDTH +
            FLT_EPSILON);
  this->number_of_CUDA_block_per_cell =
      number_of_CUDA_block_per_line * number_of_CUDA_block_per_line;

  //
  // this->super_pixel_adjacent_mat = (bool
  // *)malloc(this->super_pixel_mat_size.width *
  // this->super_pixel_mat_size.height * sizeof(bool));
  this->super_pixel_adjacent_mat = (bool *)malloc(
      this->super_pixel_mat_size.width * this->super_pixel_mat_size.height *
      this->super_pixel_mat_size.width * this->super_pixel_mat_size.height *
      sizeof(bool));

  //
  checkCudaErrors(cudaMalloc(
      (void **)&(this->dev_super_pixel_id_image),
      this->aligned_depth_size.x * this->aligned_depth_size.y * sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_super_pixel_mat),
                 this->super_pixel_mat_size.x * this->super_pixel_mat_size.y *
                     sizeof(Super_pixel)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_super_pixel_accumulate_mat),
                 this->super_pixel_mat_size.x * this->super_pixel_mat_size.y *
                     sizeof(Super_pixel)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_cell_hessain_uppers),
                 this->super_pixel_mat_size.x * this->super_pixel_mat_size.y *
                     sizeof(My_Type::Vector3f)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_cell_nabla),
                 this->super_pixel_mat_size.x * this->super_pixel_mat_size.y *
                     sizeof(My_Type::Vector2f)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_base_vectors),
                 this->super_pixel_mat_size.x * this->super_pixel_mat_size.y *
                     sizeof(Plane_coordinate)));

#if (TEST_OLD_METOD)
  //
  this->hist_mat =
      (float *)malloc(HISTOGRAM_WIDTH * HISTOGRAM_WIDTH * sizeof(float));

  //
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_hist_mat),
                 HISTOGRAM_WIDTH * HISTOGRAM_WIDTH * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_hist_normals),
                             MAX_HIST_NORMALS * sizeof(Hist_normal)));
  checkCudaErrors(
      cudaMalloc((void **)&(this->dev_hist_normal_counter), sizeof(int)));
  checkCudaErrors(cudaMalloc(
      (void **)&(this->dev_prj_distance_hist),
      ceil_by_stride((int)(MAX_VALID_DEPTH_M / MIN_PLANE_DISTANCE), 256) *
          sizeof(int)));
  // Buffers for K-means iteration
  checkCudaErrors(cudaMalloc((void **)&(this->dev_plane_mean_parameters),
                             MAX_CURRENT_PLANES * sizeof(Cell_info)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_ATA_upper_buffer),
                             3 * MAX_CURRENT_PLANES * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&(this->dev_ATb_buffer),
                             2 * MAX_CURRENT_PLANES * sizeof(float)));
#endif
}

//
void Plane_super_pixel_detector::prepare_to_detect() {
  //
  Plane_detector::prepare_to_detect();

#if (TEST_OLD_METOD)
  //
  memset(this->hist_mat, 0x00,
         HISTOGRAM_WIDTH * HISTOGRAM_WIDTH * sizeof(float));

  //
  checkCudaErrors(
      cudaMemset(this->dev_hist_mat, 0x00,
                 HISTOGRAM_WIDTH * HISTOGRAM_WIDTH * sizeof(float)));
  checkCudaErrors(cudaMemset(this->dev_hist_normals, 0x00,
                             MAX_HIST_NORMALS * sizeof(Hist_normal)));
  checkCudaErrors(cudaMemset(this->dev_hist_normal_counter, 0x00, sizeof(int)));
#endif

  //
  memset(this->super_pixel_adjacent_mat, 0x00,
         this->super_pixel_mat_size.width * this->super_pixel_mat_size.height *
             this->super_pixel_mat_size.width *
             this->super_pixel_mat_size.height * sizeof(bool));
  //
  checkCudaErrors(
      cudaMemset(this->dev_super_pixel_mat, 0x00,
                 this->super_pixel_mat_size.x * this->super_pixel_mat_size.y *
                     sizeof(Super_pixel)));
}

//
void Plane_super_pixel_detector::presegment_to_cell(
    const My_Type::Vector3f *dev_current_points,
    const My_Type::Vector3f *dev_current_normals) {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  // Init super dev_super_pixel_id_image
  {
    thread_rect.x = SUPER_PIXEL_BLOCK_WIDTH;
    thread_rect.y = SUPER_PIXEL_BLOCK_WIDTH;
    thread_rect.z = 1;
    block_rect.x = this->aligned_depth_size.x / SUPER_PIXEL_BLOCK_WIDTH;
    block_rect.y = this->aligned_depth_size.y / SUPER_PIXEL_BLOCK_WIDTH;
    block_rect.z = this->number_of_CUDA_block_per_cell;
    init_super_pixel_image_CUDA(
        block_rect, thread_rect, dev_current_points, dev_current_normals,
        this->dev_super_pixel_id_image,
        SLAM_system_settings::instance()->presegment_cell_width,
        this->number_of_CUDA_block_per_line);
    // CUDA_CKECK_KERNEL;
  }

  //
  for (int i = 0; i < 5; i++) {
    checkCudaErrors(
        cudaMemset(this->dev_super_pixel_accumulate_mat, 0x00,
                   this->super_pixel_mat_size.x * this->super_pixel_mat_size.y *
                       sizeof(Super_pixel)));

    // Update each super pixel center data
    thread_rect.x = SUPER_PIXEL_BLOCK_WIDTH;
    thread_rect.y = SUPER_PIXEL_BLOCK_WIDTH;
    thread_rect.z = 1;
    block_rect.x = this->aligned_depth_size.x / SUPER_PIXEL_BLOCK_WIDTH;
    block_rect.y = this->aligned_depth_size.y / SUPER_PIXEL_BLOCK_WIDTH;
    block_rect.z = this->number_of_CUDA_block_per_cell;
    update_cluster_center_CUDA(
        block_rect, thread_rect, dev_current_points, dev_current_normals,
        this->dev_super_pixel_id_image, this->dev_super_pixel_accumulate_mat,
        this->dev_super_pixel_mat,
        SLAM_system_settings::instance()->presegment_cell_width,
        this->number_of_CUDA_block_per_line);
    // CUDA_CKECK_KERNEL;

    // Find association for each pixel
    thread_rect.x =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.y =
        SLAM_system_settings::instance()->image_alginment_patch_width;
    thread_rect.z = 1;
    block_rect.x = this->aligned_depth_size.x / thread_rect.x;
    block_rect.y = this->aligned_depth_size.y / thread_rect.y;
    block_rect.z = 1;
    pixel_find_associate_center_CUDA(
        block_rect, thread_rect, dev_current_points, dev_current_normals,
        this->dev_super_pixel_mat, this->dev_super_pixel_id_image,
        SLAM_system_settings::instance()->presegment_cell_width,
        SLAM_system_settings::instance()->pixel_data_weight,
        SLAM_system_settings::instance()->normal_position_weight,
        SLAM_system_settings::instance()->sensor_params);
    // CUDA_CKECK_KERNEL;
  }

  // for render
  checkCudaErrors(cudaMemcpy(this->dev_current_cell_labels,
                             this->dev_super_pixel_id_image,
                             this->aligned_depth_size.width *
                                 this->aligned_depth_size.height * sizeof(int),
                             cudaMemcpyDeviceToDevice));
}

//
void Plane_super_pixel_detector::fit_plane_for_each_cell(
    const My_Type::Vector3f *dev_current_points,
    const My_Type::Vector3f *dev_current_normals,
    Cell_info *dev_cell_info_mat) {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  // Fit plane for each cell
  {
    checkCudaErrors(
        cudaMemset(this->dev_super_pixel_accumulate_mat, 0x00,
                   this->super_pixel_mat_size.x * this->super_pixel_mat_size.y *
                       sizeof(Super_pixel)));
    checkCudaErrors(
        cudaMemset(this->dev_cell_hessain_uppers, 0x00,
                   this->super_pixel_mat_size.x * this->super_pixel_mat_size.y *
                       sizeof(My_Type::Vector3f)));
    checkCudaErrors(
        cudaMemset(this->dev_cell_nabla, 0x00,
                   this->super_pixel_mat_size.x * this->super_pixel_mat_size.y *
                       sizeof(My_Type::Vector2f)));

    // Update each super pixel center data
    thread_rect.x = SUPER_PIXEL_BLOCK_WIDTH;
    thread_rect.y = SUPER_PIXEL_BLOCK_WIDTH;
    thread_rect.z = 1;
    block_rect.x = this->aligned_depth_size.x / SUPER_PIXEL_BLOCK_WIDTH;
    block_rect.y = this->aligned_depth_size.y / SUPER_PIXEL_BLOCK_WIDTH;
    block_rect.z = this->number_of_CUDA_block_per_cell;
    fit_plane_for_cells_CUDA(
        block_rect, thread_rect, dev_current_points,
        this->dev_super_pixel_id_image, this->dev_base_vectors,
        this->dev_super_pixel_mat, this->dev_super_pixel_accumulate_mat,
        this->dev_cell_hessain_uppers, this->dev_cell_nabla,
        SLAM_system_settings::instance()->presegment_cell_width,
        this->number_of_CUDA_block_per_line);
    // CUDA_CKECK_KERNEL;

    // ToDo : eliminate outlayers
  }

  // Generate cell info
  {
    // dev_cell_info_mat already initiated in
    // Plane_detector::prepare_to_detect()

    // Update each super pixel center data
    thread_rect.x = 1;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x = this->aligned_depth_size.x / SUPER_PIXEL_BLOCK_WIDTH;
    block_rect.y = this->aligned_depth_size.y / SUPER_PIXEL_BLOCK_WIDTH;
    block_rect.z = 1;
    generate_cells_info_CUDA(
        block_rect, thread_rect, this->dev_super_pixel_mat,
        SLAM_system_settings::instance()->sensor_params,
        SLAM_system_settings::instance()->presegment_cell_width,
        this->dev_cell_info_mat);
    // CUDA_CKECK_KERNEL;
  }
}

#pragma region(Flood fill)
//
inline void find_next_seed_point(Cell_info *cell_mat,
                                 const My_Type::Vector2i &cell_mat_size,
                                 int &seed_point_index) {
  int max_index = cell_mat_size.width * cell_mat_size.height;
  while (seed_point_index < max_index &&
         (cell_mat[seed_point_index].plane_index != 0 ||
          (!cell_mat[seed_point_index].is_valid_cell))) {
    seed_point_index++;
  }
  // printf("%d\n", seed_point_index);
}
//
inline bool flood_fill_search_pixel(Cell_info *cell_mat,
                                    const My_Type::Vector2i &cell_mat_size,
                                    const Plane_info &plane_params,
                                    My_Type::Vector2i &center_pixel,
                                    My_Type::Vector2i &searched_pixel) {
  bool is_find_pixel = false;

  const int offset_list[8][2] = {{-1, -1}, {0, -1},  {+1, -1}, {-1, 0},
                                 {+1, 0},  {-1, +1}, {0, +1},  {+1, +1}};

  //
  int center_index = center_pixel.x + center_pixel.y * cell_mat_size.width;
  for (int offset_index = 0; offset_index < 8; offset_index++) {
    int u_cell = offset_list[offset_index][0] + center_pixel.x;
    int v_cell = offset_list[offset_index][1] + center_pixel.y;
    // Validate index
    if (u_cell < 0 || u_cell >= cell_mat_size.width || v_cell < 0 ||
        v_cell >= cell_mat_size.height)
      continue;
    int index_cell = u_cell + v_cell * cell_mat_size.width;

    // Load cell info
    Cell_info temp_cell = cell_mat[index_cell];
    if (temp_cell.plane_index != 0 || temp_cell.counter < 50 ||
        !temp_cell.is_valid_cell)
      continue;

    // Check normal
    My_Type::Vector3f cell_normal(temp_cell.nx, temp_cell.ny, temp_cell.nz);
    My_Type::Vector3f plane_normal(plane_params.nx, plane_params.ny,
                                   plane_params.nz);
    if (cell_normal.dot(plane_normal) < 0.95) continue;

    // Check position
    My_Type::Vector3f cell_position(temp_cell.x, temp_cell.y, temp_cell.z);
    if (fabsf(fabsf(cell_position.dot(plane_normal)) - plane_params.d) > 0.04)
      continue;

    //
    is_find_pixel = true;
    searched_pixel = My_Type::Vector2i(u_cell, v_cell);
    break;
  }
  return is_find_pixel;
}
//
inline void update_plane_params(Cell_info *cell_info_mat,
                                std::vector<int> &cell_index_list,
                                Plane_info &temp_plane) {
  // Normal
  {
    int index_list_id = cell_index_list.size() - 1;
    int cell_index = cell_index_list[index_list_id];
    My_Type::Vector3f plane_normal(temp_plane.nx, temp_plane.ny, temp_plane.nz);
    My_Type::Vector3f cell_normal(cell_info_mat[cell_index].nx,
                                  cell_info_mat[cell_index].ny,
                                  cell_info_mat[cell_index].nz);
    //
    float weight =
        (float)cell_info_mat[cell_index].counter /
        ((float)cell_info_mat[cell_index].counter + (float)temp_plane.weight);
    plane_normal = plane_normal * (1.0f - weight) + (weight * cell_normal);
    plane_normal.normlize();
    temp_plane.nx = plane_normal.x;
    temp_plane.ny = plane_normal.y;
    temp_plane.nz = plane_normal.z;
    temp_plane.weight += cell_info_mat[cell_index].counter;
  }

  // Distance
  float distance = 0.0f, weight = 0.0f;
  for (int index_list_id = 0; index_list_id < cell_index_list.size();
       index_list_id++) {
    Cell_info temp_cell = cell_info_mat[cell_index_list[index_list_id]];
    My_Type::Vector3f position(temp_cell.x, temp_cell.y, temp_cell.z);
    My_Type::Vector3f plane_normal(temp_plane.nx, temp_plane.ny, temp_plane.nz);

    distance += fabsf(position.dot(plane_normal)) * (float)temp_cell.area;
    weight += temp_cell.area;
  }
  temp_plane.d = distance / weight;
}

// Build adjacent matrix
void build_adjacent_matrix_for_regions(Cell_info *cell_info_mat,
                                       My_Type::Vector2i cell_mat_size,
                                       bool *adjacent_mat,
                                       int number_of_regions) {
  // const int search_window_size = 12;
  // const int search_window[search_window_size][2] =
  //{						{ 0, -2 },
  //			{ -1, -1 }, { 0, -1 }, { +1, -1 },
  //  { -2, 0 },{ -1, 0 },			   { +1, 0 }, { +2, 0 },
  //			{ -1, +1 }, { 0, +1 }, { +1, +1 },
  //						{ 0, +2 } };

  const int search_window_size = 5;
  const int search_window[search_window_size][2] = {
      {+1, 0}, {+2, 0}, {0, +1}, {+1, +1}, {0, +2}};

  // const int search_window_size = 8;
  // const int search_window[search_window_size][2] =
  //{
  //	{ -1, -1 }, { 0, -1 }, { +1, -1 },
  //	{ -1, 0 },			   { +1, 0 },
  //	{ -1, +1 }, { 0, +1 }, { +1, +1 }
  //};

  // const int search_window_size = 3;
  // const int search_window[search_window_size][2] =
  //{
  //				{ +1, 0 },
  //	 { 0, +1 }, { +1, +1 }
  //};

  memset(adjacent_mat, 0x00,
         number_of_regions * number_of_regions * sizeof(bool));
  //
  for (int v_cell = 0; v_cell < cell_mat_size.height; v_cell++)
    for (int u_cell = 0; u_cell < cell_mat_size.width; u_cell++) {
      int center_index = u_cell + v_cell * cell_mat_size.width;
      if (!cell_info_mat[center_index].is_valid_cell) continue;
      int center_plane_id = cell_info_mat[center_index].plane_index;
      if (center_plane_id == 0) continue;
      //
      for (int check_i = 0; check_i < search_window_size; check_i++) {
        int u_check = u_cell + search_window[check_i][0];
        int v_check = v_cell + search_window[check_i][1];
        if (u_check < 0 || u_check >= cell_mat_size.width || v_check < 0 ||
            v_check >= cell_mat_size.height)
          continue;
        //
        int check_index = u_check + v_check * cell_mat_size.width;
        if (!cell_info_mat[check_index].is_valid_cell) continue;
        int check_plane_id = cell_info_mat[check_index].plane_index;
        if (check_plane_id == 0) continue;
        if (center_plane_id == check_plane_id) continue;
        //
        adjacent_mat[center_plane_id + check_plane_id * number_of_regions] =
            true;
        adjacent_mat[check_plane_id + center_plane_id * number_of_regions] =
            true;
      }
    }
}

//
void generate_local_coordinate_host(Plane_coordinate &local_coordinate) {
  local_coordinate.x_vec = My_Type::Vector3f(1.0f, 0.0f, 0.0f);
  if (fabsf(local_coordinate.x_vec.dot(local_coordinate.z_vec)) >= 0.71) {
    local_coordinate.x_vec = My_Type::Vector3f(0.0f, 1.0f, 0.0f);
    if (fabsf(local_coordinate.x_vec.dot(local_coordinate.z_vec)) >= 0.71) {
      local_coordinate.x_vec = My_Type::Vector3f(0.0f, 0.0f, 1.0f);
    }
  }
  // Orthogonalization
  local_coordinate.x_vec = local_coordinate.x_vec -
                           local_coordinate.x_vec.dot(local_coordinate.z_vec) *
                               local_coordinate.z_vec;
  // Normalize base_x
  local_coordinate.x_vec.normlize();
  //
  local_coordinate.y_vec = local_coordinate.z_vec.cross(local_coordinate.x_vec);
}
//
inline void transform_to_local_coordinate_host(
    My_Type::Vector3f &src_point, My_Type::Vector3f &dst_point,
    Plane_coordinate &local_coordinate) {
  dst_point.x = local_coordinate.x_vec.dot(src_point);
  dst_point.y = local_coordinate.y_vec.dot(src_point);
  dst_point.z = local_coordinate.z_vec.dot(src_point);
}
//
inline void transform_from_local_coordinate_host(
    My_Type::Vector3f &src_point, My_Type::Vector3f &dst_point,
    Plane_coordinate &local_coordinate) {
  dst_point = src_point.x * local_coordinate.x_vec +
              src_point.y * local_coordinate.y_vec +
              src_point.z * local_coordinate.z_vec;
}
// Fit plane for each region
void fit_params_for_planar_region(Cell_info *cell_info_mat,
                                  Plane_info &plane_region_params,
                                  std::vector<int> &planar_cell_index_list) {
  Plane_coordinate plane_coordinate;
  plane_coordinate.z_vec = My_Type::Vector3f(
      plane_region_params.nx, plane_region_params.ny, plane_region_params.nz);
  generate_local_coordinate_host(plane_coordinate);

  // Compute mean cell position
  My_Type::Vector3f mean_position(0.0f);
  float total_area = 0.0f;
  int number_of_region_cells = planar_cell_index_list.size();
  int valid_cell_counter = 0;
  for (int index_id = 0; index_id < number_of_region_cells; index_id++) {
    int cell_index = planar_cell_index_list[index_id];
    if (cell_info_mat[cell_index].is_valid_cell) {
      mean_position += My_Type::Vector3f(cell_info_mat[cell_index].x,
                                         cell_info_mat[cell_index].y,
                                         cell_info_mat[cell_index].z);
      total_area += cell_info_mat[cell_index].area;
      valid_cell_counter++;
    }
  }
  mean_position /= valid_cell_counter;
  plane_region_params.x = mean_position.x;
  plane_region_params.y = mean_position.y;
  plane_region_params.z = mean_position.z;
  plane_region_params.area = total_area;

  //
  if (valid_cell_counter >= 10) {
    //
    float hessain_upper[3], nabla[2];
    hessain_upper[0] = 0;
    hessain_upper[1] = 0;
    hessain_upper[2] = 0;
    nabla[0] = 0;
    nabla[1] = 0;
    for (int index_id = 0; index_id < number_of_region_cells; index_id++) {
      int cell_index = planar_cell_index_list[index_id];
      if (!cell_info_mat[cell_index].is_valid_cell) continue;

      My_Type::Vector3f cell_position(cell_info_mat[cell_index].x,
                                      cell_info_mat[cell_index].y,
                                      cell_info_mat[cell_index].z);
      cell_position -= mean_position;
      My_Type::Vector3f local_cell_position;
      transform_to_local_coordinate_host(cell_position, local_cell_position,
                                         plane_coordinate);

      //
      hessain_upper[0] += local_cell_position.x * local_cell_position.x;
      hessain_upper[1] += local_cell_position.x * local_cell_position.y;
      hessain_upper[2] += local_cell_position.y * local_cell_position.y;
      nabla[0] += -local_cell_position.x * local_cell_position.z;
      nabla[1] += -local_cell_position.y * local_cell_position.z;
    }

    // Solve
    float D, D1, D2, tan_xz, tan_yz;
    // Compute Crammer
    D = hessain_upper[0] * hessain_upper[2] -
        hessain_upper[1] * hessain_upper[1];
    if (D > 1.0f) {
      D1 = nabla[0] * hessain_upper[2] - nabla[1] * hessain_upper[1];
      D2 = nabla[1] * hessain_upper[0] - nabla[0] * hessain_upper[1];
      // compute tangent
      tan_xz = D1 / D;
      tan_yz = D2 / D;
      // printf("%f\n", D);
      //
      My_Type::Vector3f normal_vec_local, plane_normal_vec;
      normal_vec_local.z = 1 / My_Type::Vector3f(tan_xz, tan_yz, 1.0f).norm();
      normal_vec_local.x = tan_xz * normal_vec_local.z;
      normal_vec_local.y = tan_yz * normal_vec_local.z;
      normal_vec_local.normlize();
      //
      transform_from_local_coordinate_host(normal_vec_local, plane_normal_vec,
                                           plane_coordinate);
      plane_normal_vec.normlize();
      // Check normal direction
      if (mean_position.dot(plane_normal_vec) > 0) {
        plane_normal_vec.nx = -plane_normal_vec.nx;
        plane_normal_vec.ny = -plane_normal_vec.ny;
        plane_normal_vec.nz = -plane_normal_vec.nz;
      }

      plane_region_params.nx = plane_normal_vec.nx;
      plane_region_params.ny = plane_normal_vec.ny;
      plane_region_params.nz = plane_normal_vec.nz;

#pragma region(Compute distance)
      float distance = 0;
      for (int index_id = 0; index_id < number_of_region_cells; index_id++) {
        int cell_index = planar_cell_index_list[index_id];
        if (!cell_info_mat[cell_index].is_valid_cell) continue;

        My_Type::Vector3f cell_position(cell_info_mat[cell_index].x,
                                        cell_info_mat[cell_index].y,
                                        cell_info_mat[cell_index].z);

        distance += fabsf(cell_position.dot(plane_normal_vec));
      }
      distance /= number_of_region_cells;

      plane_region_params.d = distance;
#pragma endregion

      plane_region_params.is_valid = true;

      return;
    }
  }

  if (valid_cell_counter > 0) {
    // Compute mean normal
    My_Type::Vector3f mean_normal(0.0f);
    for (int index_id = 0; index_id < number_of_region_cells; index_id++) {
      int cell_index = planar_cell_index_list[index_id];
      if (cell_info_mat[cell_index].is_valid_cell)
        mean_normal += My_Type::Vector3f(cell_info_mat[cell_index].nx,
                                         cell_info_mat[cell_index].ny,
                                         cell_info_mat[cell_index].nz);
    }
    mean_normal /= valid_cell_counter;
    mean_normal.normlize();
    //
    plane_region_params.nx = mean_normal.nx;
    plane_region_params.ny = mean_normal.ny;
    plane_region_params.nz = mean_normal.nz;

    //
    plane_region_params.d = fabsf(mean_normal.dot(mean_position));

    plane_region_params.is_valid = true;
  } else {
    plane_region_params.is_valid = false;
  }
}

//
bool search_merge_region(int &base_id, int &search_id,
                         std::vector<bool> &merge_flag,
                         std::vector<Plane_info> &plane_region_params,
                         bool *adjacent_mat) {
  int number_of_regions = plane_region_params.size();
  for (search_id = 0; search_id < number_of_regions; search_id++) {
    if (adjacent_mat[base_id * number_of_regions + search_id] == true &&
        merge_flag[search_id] == false && search_id != base_id &&
        plane_region_params[search_id].is_valid) {
      My_Type::Vector3f normal_base(plane_region_params[base_id].nx,
                                    plane_region_params[base_id].ny,
                                    plane_region_params[base_id].nz);
      My_Type::Vector3f normal_search(plane_region_params[search_id].nx,
                                      plane_region_params[search_id].ny,
                                      plane_region_params[search_id].nz);
      const float normal_threshold = 0.95f;
      float inner_product = normal_base.dot(normal_search);
      if (inner_product < normal_threshold) continue;
      // TODO
      const float distance_threshold = 0.04f;
      My_Type::Vector3f plane_center(plane_region_params[search_id].x,
                                     plane_region_params[search_id].y,
                                     plane_region_params[search_id].z);
      My_Type::Vector3f base_center(plane_region_params[base_id].x,
                                    plane_region_params[base_id].y,
                                    plane_region_params[base_id].z);

      float diff_dist = fabsf((plane_center - base_center).dot(normal_base));
      // printf("(%f, %f, %f) (%f, %f, %f)\n",
      //	   plane_center.x, plane_center.y, plane_center.z,
      //	   normal_base.x, normal_base.y, normal_base.z);
      // printf("%f, %f\n", inner_product, diff_dist);
      if (diff_dist > distance_threshold) continue;

      return true;
    }
  }
  return false;
}

//
void merge_planar_regions(
    Cell_info *cell_info_mat, My_Type::Vector2i cell_mat_size,
    bool *adjacent_mat, std::vector<Plane_info> &plane_region_params,
    std::vector<std::vector<int>> &planar_cell_index_list) {
  std::vector<bool> merge_flag(plane_region_params.size(), false);
  std::vector<std::vector<int>> merge_list;
  merge_list.push_back(std::vector<int>());

  int seed_id = 1;
  while (seed_id < plane_region_params.size()) {
    std::vector<int> merge_region;
    std::vector<int> region_index_stack;

    //
    region_index_stack.push_back(seed_id);
    merge_flag[seed_id] = true;
    merge_region.push_back(seed_id);

    do {
      int base_id, searched_id;
      base_id = region_index_stack.back();
      bool is_find = search_merge_region(base_id, searched_id, merge_flag,
                                         plane_region_params, adjacent_mat);
      if (is_find) {
        //
        region_index_stack.push_back(searched_id);
        merge_flag[searched_id] = true;
        merge_region.push_back(searched_id);
      } else {
        region_index_stack.pop_back();
      }
    } while (region_index_stack.size() > 0);
    merge_list.push_back(merge_region);

    // Find next seed id
    while (seed_id < plane_region_params.size() &&
           (merge_flag[seed_id] || (!plane_region_params[seed_id].is_valid))) {
      merge_flag[seed_id] = true;
      seed_id++;
    }
  }

  for (int cell_index = 0;
       cell_index < cell_mat_size.width * cell_mat_size.height; cell_index++)
    cell_info_mat[cell_index].plane_index = 0;

  //
  std::vector<Plane_info> plane_param_buffer;
  plane_param_buffer.push_back(Plane_info());
  std::vector<std::vector<int>> planar_cell_index_list_buffer;
  planar_cell_index_list_buffer.push_back(std::vector<int>());
  //
  for (int new_plane_id = 1; new_plane_id < merge_list.size(); new_plane_id++) {
    std::vector<int> temp_cell_index_list;
    for (int j = 0; j < merge_list[new_plane_id].size(); j++) {
      int list_id = merge_list[new_plane_id][j];

      for (int cell_index_id = 0;
           cell_index_id < planar_cell_index_list[list_id].size();
           cell_index_id++) {
        int cell_index = planar_cell_index_list[list_id][cell_index_id];
        cell_info_mat[cell_index].plane_index = new_plane_id;
      }
      temp_cell_index_list.insert(temp_cell_index_list.end(),
                                  planar_cell_index_list[list_id].begin(),
                                  planar_cell_index_list[list_id].end());
    }
    // Update merged plane info
    Plane_info temp_plane_info;
    fit_params_for_planar_region(cell_info_mat, temp_plane_info,
                                 temp_cell_index_list);
    plane_param_buffer.push_back(temp_plane_info);
    // printf("%d : %f, %f, %f, %f", new_plane_id , temp_plane_info.nx,
    // temp_plane_info.ny, temp_plane_info.nz, temp_plane_info.d);
    // printf("\t--%d\n", temp_cell_index_list.size());
    planar_cell_index_list_buffer.push_back(temp_cell_index_list);
  }

  plane_region_params.clear();
  plane_region_params = plane_param_buffer;
  planar_cell_index_list = planar_cell_index_list_buffer;
}

// Set large region as plane
void relabel_cells_by_large_regions(
    Cell_info *cell_info_mat, std::vector<Plane_info> &plane_region_params,
    std::vector<std::vector<int>> &planar_cell_index_list) {
  std::vector<Plane_info> plane_param_buffer;
  plane_param_buffer.push_back(Plane_info());

  std::vector<std::vector<int>> planar_cell_index_list_buffer;
  planar_cell_index_list_buffer.push_back(std::vector<int>());

  //
  int new_plane_id = 1;
  for (int plane_id = 1, new_plane_id = 1;
       plane_id < plane_region_params.size(); plane_id++) {
    float plane_area = plane_region_params[plane_id].area;

    //
    const float region_area_threshold = MIN_AREA_OF_PLANE;
    if (plane_area < region_area_threshold ||
        (!plane_region_params[plane_id].is_valid)) {
      for (int cell_index_id = 0;
           cell_index_id < planar_cell_index_list[plane_id].size();
           cell_index_id++) {
        int cell_index = planar_cell_index_list[plane_id][cell_index_id];
        cell_info_mat[cell_index].plane_index = 0;
      }
    } else {
      for (int cell_index_id = 0;
           cell_index_id < planar_cell_index_list[plane_id].size();
           cell_index_id++) {
        int cell_index = planar_cell_index_list[plane_id][cell_index_id];
        cell_info_mat[cell_index].plane_index = new_plane_id;
      }

      plane_param_buffer.push_back(plane_region_params[plane_id]);
      new_plane_id++;
      // printf("(%f, %f, %f) %f ; \tA = ",
      //	   plane_region_params[plane_id].nx,
      //	   plane_region_params[plane_id].ny,
      //	   plane_region_params[plane_id].nz,
      //	   plane_region_params[plane_id].d);
      // printf("%f\n",  plane_area);
      planar_cell_index_list_buffer.push_back(planar_cell_index_list[plane_id]);
    }
  }

  plane_region_params.clear();
  plane_region_params = plane_param_buffer;
  planar_cell_index_list.clear();
  planar_cell_index_list = planar_cell_index_list_buffer;
}

#pragma endregion

//
void Plane_super_pixel_detector::cluster_cells(
    Cell_info *dev_cell_info_mat, const Plane_info *dev_model_planes,
    Plane_info *dev_current_planes, bool with_continuous_frame_tracking) {
  dim3 block_rect(1, 1, 1), thread_rect(1, 1, 1);

  // Copy out cell infomation
  checkCudaErrors(cudaMemcpy(this->cell_info_mat, this->dev_cell_info_mat,
                             this->cell_mat_size.width *
                                 this->cell_mat_size.height * sizeof(Cell_info),
                             cudaMemcpyDeviceToHost));

  // Flood fill
#pragma region(Flood fill)
  //
  {
    //// Init
    // for (int cell_index = 0; cell_index < this->cell_mat_size.width *
    // this->cell_mat_size.height; cell_index++)
    //	this->cell_info_mat[cell_index].plane_index = 0;

    // Merge SuperPixels ------------------
    //
    int seed_point_index = 0, temp_plane_index = 0;
    int cell_mat_length =
        this->cell_mat_size.width * this->cell_mat_size.height;
    //
    this->plane_region_params.clear();
    this->plane_region_params.push_back(Plane_info());
    this->planar_cell_index_list.clear();
    this->planar_cell_index_list.push_back(std::vector<int>());
    //
    while (seed_point_index < cell_mat_length) {
      temp_plane_index++;
      //
      Plane_info temp_plane(0, 0, 0, 0, 0, 0, 0, 0);
      std::vector<int> cell_index_list;
      std::vector<My_Type::Vector2i> point_stack;

      // Init plane params
      find_next_seed_point(this->cell_info_mat, this->cell_mat_size,
                           seed_point_index);
      cell_index_list.push_back(seed_point_index);
      update_plane_params(this->cell_info_mat, cell_index_list, temp_plane);
      find_next_seed_point(this->cell_info_mat, this->cell_mat_size,
                           seed_point_index);

      //
      My_Type::Vector2i center_pixel(
          seed_point_index % this->cell_mat_size.width,
          seed_point_index / this->cell_mat_size.width);
      cell_info_mat[seed_point_index].plane_index = temp_plane_index;
      point_stack.push_back(center_pixel);

      //
      do {
        My_Type::Vector2i searched_pixel;
        bool find_next =
            flood_fill_search_pixel(this->cell_info_mat, this->cell_mat_size,
                                    temp_plane, center_pixel, searched_pixel);
        // Update plane_params and cell_mat
        if (find_next) {
          // Update center
          center_pixel = searched_pixel;
          int center_index =
              center_pixel.x + center_pixel.y * this->cell_mat_size.width;
          this->cell_info_mat[center_index].plane_index = temp_plane_index;
          // Update plane
          cell_index_list.push_back(center_index);
          point_stack.push_back(center_pixel);
          update_plane_params(this->cell_info_mat, cell_index_list, temp_plane);
          //
          find_next_seed_point(this->cell_info_mat, this->cell_mat_size,
                               seed_point_index);
        } else {
          point_stack.pop_back();
          if (point_stack.size() > 0) {
            searched_pixel = point_stack.back();
          }
        }
        // printf("point_stack.size() = %d\n", point_stack.size());
      } while (point_stack.size() > 0);

      if (cell_index_list.size() > 0) {
        this->planar_cell_index_list.push_back(cell_index_list);
        this->plane_region_params.push_back(temp_plane);
      }
    }

    // give plane label to cell
    int number_of_regions = (int)this->plane_region_params.size();
    for (int list_id = 1; list_id < number_of_regions; list_id++) {
      for (int cell_index_id = 0;
           cell_index_id < this->planar_cell_index_list[list_id].size();
           cell_index_id++) {
        int index = (this->planar_cell_index_list[list_id])[cell_index_id];
        this->cell_info_mat[index].plane_index = list_id;
      }
    }

    // Merge regions ------------------
    // Build adjacent matrix
    build_adjacent_matrix_for_regions(this->cell_info_mat, this->cell_mat_size,
                                      this->super_pixel_adjacent_mat,
                                      number_of_regions);

    //
    for (int i = 1; i < number_of_regions; i++)
      fit_params_for_planar_region(this->cell_info_mat,
                                   this->plane_region_params[i],
                                   this->planar_cell_index_list[i]);

    // Merge
    merge_planar_regions(this->cell_info_mat, this->cell_mat_size,
                         this->super_pixel_adjacent_mat,
                         this->plane_region_params,
                         this->planar_cell_index_list);

    // Region to plane
    relabel_cells_by_large_regions(this->cell_info_mat,
                                   this->plane_region_params,
                                   this->planar_cell_index_list);

    // Fit plane
    for (int i = 1; i < this->planar_cell_index_list.size(); i++)
      fit_params_for_planar_region(this->cell_info_mat,
                                   this->plane_region_params[i],
                                   this->planar_cell_index_list[i]);

    //// For debug
    // number_of_regions = (int)this->plane_region_params.size();
    // build_adjacent_matrix_for_regions(this->cell_info_mat,
    // this->cell_mat_size, this->super_pixel_adjacent_mat, number_of_regions);

    checkCudaErrors(
        cudaMemcpy(this->dev_cell_info_mat, this->cell_info_mat,
                   this->cell_mat_size.width * this->cell_mat_size.height *
                       sizeof(Cell_info),
                   cudaMemcpyHostToDevice));

    //
    this->current_plane_counter = this->plane_region_params.size();
    for (int plane_id = 1; plane_id < this->current_plane_counter; plane_id++)
      this->current_planes[plane_id] = this->plane_region_params[plane_id];
  }
#pragma endregion

  // Old method
#if (TEST_OLD_METOD)
#pragma region(Old)

#pragma region(Generate plane paramter seed)
  // 1. Histogram normal direction of cells on PxPy(use stereoproject [nx, ny,
  // nz] to [px, py])
  {
    thread_rect.x = 256;
    thread_rect.y = 1;
    thread_rect.z = 1;
    block_rect.x =
        this->cell_mat_size.width * this->cell_mat_size.height / thread_rect.x;
    block_rect.y = 1;
    block_rect.z = 1;
    // Lunch kernel function
    histogram_PxPy_CUDA(block_rect, thread_rect, dev_cell_info_mat,
                        this->dev_hist_mat);
    // CUDA_CKECK_KERNEL;

    // Copy out histogram
    checkCudaErrors(
        cudaMemcpy(this->hist_mat, this->dev_hist_mat,
                   HISTOGRAM_WIDTH * HISTOGRAM_WIDTH * sizeof(float),
                   cudaMemcpyDeviceToHost));
  }

  // 2. Search peaks on PxPy as plane normal direction
  {
    // Detect plane direction from last frame
    checkCudaErrors(cudaMemset(this->dev_hist_mat, 0x00,
                               MAX_HIST_NORMALS * sizeof(Hist_normal)));
    checkCudaErrors(
        cudaMemset(this->dev_hist_normal_counter, 0x00, sizeof(int)));
    //
    thread_rect.x = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.y = SLAM_system_settings::instance()->presegment_cell_width;
    thread_rect.z = 1;
    block_rect.x = HISTOGRAM_WIDTH / thread_rect.x;
    block_rect.y = HISTOGRAM_WIDTH / thread_rect.y;
    block_rect.z = 1;
    // Lunch kernel function
    find_PxPy_peaks_CUDA(block_rect, thread_rect, this->dev_hist_mat,
                         this->dev_hist_normals, this->dev_hist_normal_counter);
    // CUDA_CKECK_KERNEL;
    // Copy out peaks
    checkCudaErrors(cudaMemcpy(&this->hist_normal_counter,
                               this->dev_hist_normal_counter, sizeof(int),
                               cudaMemcpyDeviceToHost));
  }

  // 3. Find planes in each direction
  {
    int init_counter = 1;
    checkCudaErrors(cudaMemcpy(this->dev_current_plane_counter, &init_counter,
                               sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(this->dev_current_planes, 0x00,
                               MAX_CURRENT_PLANES * sizeof(Plane_info)));
    //
    Hist_normal *dev_hist_normal_ptr = this->dev_hist_normals;
    // printf("hist_normal_counter = %d\r\n", this->hist_normal_counter);
    for (int i = 0; i < this->hist_normal_counter; i++) {
      // Reset distance histogram
      checkCudaErrors(cudaMemset(
          this->dev_prj_distance_hist, 0x00,
          ceil_by_stride((int)(MAX_VALID_DEPTH_M / MIN_PLANE_DISTANCE), 256) *
              sizeof(float)));
      // Set CUDA grid size
      thread_rect.x = 256;
      thread_rect.y = 1;
      thread_rect.z = 1;
      block_rect.x =
          ceil_by_stride(this->cell_mat_size.width * this->cell_mat_size.height,
                         256) /
          thread_rect.x;
      block_rect.y = 1;
      block_rect.z = 1;
      // Lunch kernel function
      histogram_prj_dist_CUDA(block_rect, thread_rect, this->dev_cell_info_mat,
                              dev_hist_normal_ptr, this->dev_prj_distance_hist);
      // CUDA_CKECK_KERNEL;

      // Set CUDA grid size
      thread_rect.x = 256;
      thread_rect.y = 1;
      thread_rect.z = 1;
      block_rect.x =
          ceil_by_stride((int)(MAX_VALID_DEPTH_M / MIN_PLANE_DISTANCE), 256) /
          thread_rect.x;
      block_rect.y = 1;
      block_rect.z = 1;
      // Lunch kernel function
      find_prj_dist_peaks_CUDA(block_rect, thread_rect,
                               this->dev_prj_distance_hist, dev_hist_normal_ptr,
                               this->dev_current_plane_counter,
                               this->dev_current_planes);
      // CUDA_CKECK_KERNEL;

      // Next normal pointer
      dev_hist_normal_ptr++;
    }

    // Copy out current plane number
    checkCudaErrors(cudaMemcpy(&this->current_plane_counter,
                               this->dev_current_plane_counter, sizeof(int),
                               cudaMemcpyDeviceToHost));
  }
#pragma endregion

#pragma region(K - means iteration)
  // 4. K-means iteration
  {
    const int kmeans_iteration_times = 5;
    for (int i = 0; i < kmeans_iteration_times; i++) {
      // Prepare for next iteration
      checkCudaErrors(cudaMemset(this->dev_plane_mean_parameters, 0x00,
                                 MAX_CURRENT_PLANES * sizeof(Cell_info)));
      checkCudaErrors(cudaMemset(this->dev_ATA_upper_buffer, 0x00,
                                 3 * MAX_CURRENT_PLANES * sizeof(float)));
      checkCudaErrors(cudaMemset(this->dev_ATb_buffer, 0x00,
                                 2 * MAX_CURRENT_PLANES * sizeof(float)));

      // Set CUDA grid size
      thread_rect.x = 256;
      thread_rect.y = 1;
      thread_rect.z = 1;
      block_rect.x =
          ceil_by_stride(this->cell_mat_size.width * this->cell_mat_size.height,
                         thread_rect.x) /
          thread_rect.x;
      block_rect.y = 1;
      block_rect.z = 1;
      // Lunch kernel function
      K_mean_iterate_CUDA(
          block_rect, thread_rect, this->dev_current_planes,
          this->dev_cell_info_mat, this->dev_plane_mean_parameters,
          this->dev_buffer_coordinate, this->dev_ATA_upper_buffer,
          this->dev_ATb_buffer, this->current_plane_counter);
      // CUDA_CKECK_KERNEL;
    }
  }
#pragma endregion

  // Copy out current plane information
  checkCudaErrors(cudaMemcpy(&this->current_plane_counter,
                             this->dev_current_plane_counter, sizeof(int),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(this->current_planes, this->dev_current_planes,
                             this->current_plane_counter * sizeof(Plane_info),
                             cudaMemcpyDeviceToHost));

#pragma endregion
#endif

  // for (int plane_id = 1; plane_id < this->current_plane_counter; plane_id++)
  //{
  //	My_Type::Vector3f normal_vec(this->current_planes[plane_id].nx,
  // this->current_planes[plane_id].ny,
  // this->current_planes[plane_id].nz); 	printf("%d : %f, %f, %f, %f\n",
  //plane_id, 		   this->current_planes[plane_id].nx,
  // this->current_planes[plane_id].ny,
  // this->current_planes[plane_id].nz, normal_vec.norm());
  //}

  // Re-label
  {
    // Update each super pixel center data
    thread_rect.x = SUPER_PIXEL_BLOCK_WIDTH;
    thread_rect.y = SUPER_PIXEL_BLOCK_WIDTH;
    thread_rect.z = 1;
    block_rect.x = this->aligned_depth_size.x / SUPER_PIXEL_BLOCK_WIDTH;
    block_rect.y = this->aligned_depth_size.y / SUPER_PIXEL_BLOCK_WIDTH;
    block_rect.z = this->number_of_CUDA_block_per_cell;
    relabel_super_pixels_CUDA(
        block_rect, thread_rect, this->dev_cell_info_mat,
        this->dev_super_pixel_id_image, this->dev_current_plane_labels,
        SLAM_system_settings::instance()->presegment_cell_width,
        this->number_of_CUDA_block_per_line);
    // CUDA_CKECK_KERNEL;
  }
}
