/*
 * Copyright 2025, Kaiyuan Zhang
 *
 * 该功能包接收/livox/lidar/pointcloud原始点云并分离动态障碍物
 * 1. 读入先验点云地图，使用pcl将其构建为一个kd tree
 * 2. 将每一帧实时点云与先验进行对比
 * - 接收一帧新雷达点云
 * - 遍历每个点P
 * - 对于每个点P,搜索距离其最近的点P_prior
 * - 计算欧式距离的平方
 * - 如果距离小于阈值，属于静态；反之，属于动态
 * - 将动态和静态点云分别存储并发布
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h> 
#include <string>
#include <vector>
#include <omp.h> 

using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;

class DynamicRemoval : public rclcpp::Node
{
public:
  DynamicRemoval() : Node("dynamic_removal")
  {
    // TODO 
    // 声明表示先验点云文件路径的参数 (假设文件名为scans.pcd)
    // ---
    this->declare_parameter<std::string>("pcd_file_path","scans.pcd");
    // ---
    this->declare_parameter<double>("distance_threshold", 0.2);
    // 体素下采样参数
    this->declare_parameter<double>("downsample_leaf_size", 0.1);

    pcd_file_path_ = this->get_parameter("pcd_file_path").as_string();
    distance_sq_threshold_ = this->get_parameter("distance_threshold").as_double();
    distance_sq_threshold_ *= distance_sq_threshold_;
    // TODO
    // 提取体素下采样参数，注意变量名
    double downsample_leaf_size=this->get_parameter("downsample_leaf_size").as_double();

    // ---

    // 初始化体素网格滤波器
    downsample_filter_.setLeafSize(downsample_leaf_size, downsample_leaf_size, downsample_leaf_size);

    initialize();

    RCLCPP_INFO(this->get_logger(), "Dynamic Removal node initialized with optimizations.");
  }

private:
  void initialize()
  {
    prior_map_ = std::make_shared<PointCloud>();
    if (pcl::io::loadPCDFile<PointT>(pcd_file_path_, *prior_map_) == -1)
    {
      RCLCPP_ERROR(this->get_logger(), "Couldn't read prior pcd file: %s", pcd_file_path_.c_str());
      rclcpp::shutdown();
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Loaded prior map with %zu points from %s", prior_map_->size(), pcd_file_path_.c_str());

    // TODO
    // 构建搜索结构，提示: kd-tree
    kdtree_=std::make_shared<pcl::KdTreeFLANN<PointT>>();
    kdtree_->setInputCloud(prior_map_);

    // ---
    RCLCPP_INFO(this->get_logger(), "KD-Tree built successfully.");

    dynamic_cloud_ = std::make_shared<PointCloud>();
    static_cloud_ = std::make_shared<PointCloud>();

    // TODO
    // 订阅原始点云，发布动态点云和静态点云
    // ---
    sub_=this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/llvox/lidar/pointcloud",10,std::bind(&DynamicRemoval::cloud_callback,this,std::placeholders::_1)
    );
    static_pub_=this->create_publisher<sensor_msgs::msg::PointCloud2>("static_pointcloud",10);
    dynamic_pub_=this->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_pointcloud",10);
    // ---
  }

  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    PointCloud::Ptr current_cloud(new PointCloud);
    pcl::fromROSMsg(*msg, *current_cloud);

    // TODO
    // 体素下采样
    // ---
    PointCloud::Ptr downsampled_cloud(new PointCloud);
    downsample_filter_.setInputCloud(current_cloud);
    downsample_filter_.filter(*downsampled_cloud);
    
    /*漏洞：这里应该添加一个检查采样后点云是否为空
    
    if(downsampled_cloud->empty()){
      RCLCPP_WARN(this->get_logger(),"Downsampled cloud is empty!");
    }
    */
    static_cloud_->clear();
    dynamic_cloud_->clear();
    // --- 

    // TODO
    // 近邻搜索 kd-tree
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);

    #pragma omp parallel for
    for(size_t i=0;i<downsampled_cloud->size();++i){
      PointT point=downsampled_cloud->points[i];

      if(kdtree_->nearestKSearch(point,1,pointIdxNKNSearch,pointNKNSquaredDistance)>0){
        if(pointNKNSquaredDistance[0]<distance_sq_threshold_){
          #pragma omp critical
          static_cloud_->push_back(point);
        }
        else{
          #pragma omp critical
          dynamic_cloud_->push_back(point);
        }
      }
    }
    // ---

    // TODO
    // 结果发布
    // ---
   publish_clouds(msg->header);
    // ---
  }
  void publish_clouds(const std_msgs::msg::Header& header)
  {
    sensor_msgs::msg::PointCloud2 static_msg, dynamic_msg;
    
    pcl::toROSMsg(*static_cloud_, static_msg);
    static_msg.header = header;

    //点云发布时需要设置坐标系frame_id吗？坐标系会不会混乱啊？
    
    pcl::toROSMsg(*dynamic_cloud_, dynamic_msg);
    dynamic_msg.header = header; 

    static_pub_->publish(static_msg);
    dynamic_pub_->publish(dynamic_msg);
  }

  // 参数
  std::string pcd_file_path_;
  double distance_sq_threshold_;

  PointCloud::Ptr prior_map_;
  pcl::KdTreeFLANN<PointT>::Ptr kdtree_;
  PointCloud::Ptr dynamic_cloud_;
  PointCloud::Ptr static_cloud_;
  pcl::VoxelGrid<PointT> downsample_filter_; 

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dynamic_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr static_pub_;
};

int main(int argc, char *argv[]) 
{
  // TODO
  // 主函数
  rclcpp::init(argc,argv);
  auto node =std::make_shared<DynamicRemoval>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;

  // ---
}
