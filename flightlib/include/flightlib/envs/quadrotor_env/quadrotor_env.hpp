#pragma once

// std lib
#include <stdlib.h>
#include <cmath>
#include <iostream>

// standard library
#include <unistd.h>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

// alpha gym types
#include "flightlib/common/types.hpp"

// yaml cpp
#include <yaml-cpp/yaml.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/math.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include <opencv2/core/eigen.hpp>


namespace flightlib {

namespace quadenv {

enum Ctl : int {
  // observations
  kObs = 0,
  //
  kVelRef = 0,
  kNVelRef = 3,
  kOri = 3,
  kNOri = 9,
  kLinVel = 12,
  kNLinVel = 3,
  kAngVel = 15,
  kNAngVel = 3,
  kNObs = 18,
  // control actions
  kAct = 0,
  kNAct = 4,
  frame_height = 128,
  frame_width = 128,
  camera_FOV = 120, 
  // termination reasons
  termCrashInWorldBox = 1,
  termCollisionWithObstacle = 2,
  termTimeIsUp = 3, 
  velocityTooOff = 4, 
};
};
class QuadrotorEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  QuadrotorEnv();
  QuadrotorEnv(const std::string &cfg_path);
  ~QuadrotorEnv();
  

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, Ref<DepthImage<>> img, const bool random = true) override;
  bool resetObs(Ref<Vector<>> obs, const bool random = true);
  bool resetImages(Ref<DepthImage<>> img);

  void setResetPose(Vector<3> &resetPosition, Vector<3> &resetRotation);
  void setVelocityReference(Ref<Vector<>> reference_velocity_high_level_controller);
  void setQuadstateToResetState();
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;
  Scalar stepUnity(Ref<DepthImage<>> img);

  // - public set functions
  bool loadParam(const YAML::Node &cfg);

  // public variables
  std::unordered_map<std::string, float> extra_info_;

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getImages(Ref<DepthImage<>> img);
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;
  void LogRewardInfo();
  void updateExtraInfo();

  void RandomizeReference();
  void DrawRandomGoal();
  void DrawRandomVelocity();
  void CalculateVelocityReference();

  // - auxiliar functions
  bool isTerminalState(Scalar &reward) override;
  bool isTerminalStateUnity(Scalar &reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);
  void resetReference();
  void resetWorldBox();
  Scalar OnCollisionVelocityPenalty();

  friend std::ostream &operator<<(std::ostream &os,
                                  const QuadrotorEnv &quad_env);

 private:
  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"QaudrotorEnvCamera"};

  Vector<3> resetPosition_ = Vector<3>::Zero();
  Vector<3> resetRotation_ = Vector<3>::Zero();

  inline Scalar LossFunctionSqrt(Scalar error) {
    if (error==0) { return 0; }
    return - (pow((std::abs(error)/10), 0.2) * 2);
  }

  inline Scalar LossFunctionAtan(Scalar error) {
    return -atan(error) / M_PI_2;
  }

  inline Scalar Sign(Scalar val) {
    return (0 < val) - (val < 0);
  }

  inline Scalar VelocityMagnitudeScaling(Scalar &goal_distance) {
    return atan(goalDistance - switch_goal_distance_ + 0.5) / M_PI_2;
  }

  //camera
  bool rendering{false};
  bool collect_data{false};
  std::shared_ptr<RGBCamera> rgb_camera;
  int offset_data;
  int subsampling;
  int save_count;
  int step_count;
  int goal_step_count;
  int max_steps;
  int avg_steps_per_ref;
  int goal_change_steps;
  Scalar switch_goal_distance_;

  // Variables for goal based planner.
  bool high_level_controller_mode_ = false;

  // Auxiliary flags
  int verbose_level_;
  int debug_;

  // Paths
  std::string base_path;
  std::string data_path;

  std::string count_str;
  cv::Mat img_;
  cv::Mat img_depth_;
  cv::Mat img_segm_;
  cv::Mat img_flow_;
  cv::Mat img_mask_;
  cv::Mat channels[3];
  Image_mat<quadenv::frame_height, quadenv::frame_width> img_mat_[3];
  Depth_image_mat<quadenv::frame_height, quadenv::frame_width> depth_img_mat_;

  // Define reward for training
  Scalar alive_rew, ori_coeff_, lin_vel_ori_coeff_, lin_vel_still_coeff_, lin_vel_mag_coeff_, vel_off_mag, vel_off_ori;
  Scalar kp_ang_velz;
  int vel_off_steps, vel_fading_sat, vel_fading_exp;
  Vector<4> act_coeff_;
  Vector<3> ang_vel_coeff_;
  Vector<2> drone_front_axis_proj;

  // observations and actions (for RL)
  Vector<quadenv::kNObs> quad_obs_;
  Vector<quadenv::kNAct> quad_act_;
  Vector<3> init_goal_relative_position_;
  Vector<3> omega_max_;
  Vector<3> relativeGoalPosition;
  Scalar goalDistance;
  Scalar goal_range;
  Scalar init_max_speed;
  std::vector<float> init_ori_factor;

  int reference_mode_ = 0;
  int permanent_goal = 0;
  Scalar speed_ref;
  Scalar speed_ref_old;
  Scalar scaled_speed_ref;
  Scalar maxSpeed;
  Scalar speed_threshold;
  Scalar speed_close_to_goal;
  Scalar thrust_scaling;
  Scalar vel_orientation_factor;
  Vector<3> velocity_ref_inertial;
  Vector<3> velocity_ref_inertial_versor;
  Vector<2> ref_velocity_versor_proj;
  Vector<3> velocity_ref_body;
  Vector<3> velocity_ref_body_versor;
  Matrix<3,3> rot_matrix;
  Quaternion rot_inertial2body;

  // reward function design (for model-free reinforcement learning)
  Vector<quadenv::kNObs> goal_state_;

  // action and observation normalization (for learning)
  Vector<quadenv::kNAct> act_mean_;
  Vector<quadenv::kNAct> act_std_;
  Vector<quadenv::kNObs> obs_mean_ = Vector<quadenv::kNObs>::Zero();
  Vector<quadenv::kNObs> obs_std_ = Vector<quadenv::kNObs>::Ones();

  // // random variable generator
  // std::normal_distribution<Scalar> norm_dist_{0.0, 1.0};
  // std::uniform_real_distribution<Scalar> uniform_dist_{-1.0, 1.0};
  // std::random_device rd_;
  // std::mt19937 random_gen_{rd_()};

  // // // control time step
  // Scalar sim_dt_{0.02};
  // Scalar max_t_{5.0};

  YAML::Node cfg_;
  YAML::Node cfg_vec_;
  Matrix<3, 2> world_box_;
  Matrix<3, 2> world_box_evaluation_;

  Scalar crash_vel_coeff_;
  Scalar crash_dist_coeff_;
  Scalar crash_penalty_;
  Scalar world_box_penalty_;
  Scalar velocity_too_off_penalty_;
  Scalar DistanceToGoalPenalty();

  bool justHadCollision = false;
  int collision_step_count = 0;

  // Debug and extra_info
  int episode_length_ = 0;
  Vector<7> instant_reward_components_ = Vector<7>::Zero();
  Vector<7> cumulative_reward_components_ = Vector<7>::Zero();
  Vector<7> cumulative_reward_ = Vector<7>::Zero();
  Scalar terminal_reward_ = 0;
  Scalar terminal_reason_ = 0;
  std::string pos_rew_key_ = "pos_rew";
  // std::string tube_rew_key_ = "tube_rew";
  std::string ori_rew_key_          = "ori_rew";
  std::string lin_vel_ori_rew_key_  = "lin_vel_ori_rew";
  std::string lin_vel_mag_rew_key_  = "lin_vel_mag_rew";
  std::string ang_vel_rew_key_      = "ang_vel_rew";
  std::string act_rew_key_          = "act_rew";
  std::string survival_rew_key_     = "survival_rew";
  std::string term_rew_key_         = "term_rew";
  std::string term_reason_key_      = "term_reason";
  std::string relative_pos_x_key_ = "relative_pos_x";
  std::string relative_pos_y_key_ = "relative_pos_y";
  std::string relative_pos_z_key_ = "relative_pos_z";
};

}  // namespace flightlib