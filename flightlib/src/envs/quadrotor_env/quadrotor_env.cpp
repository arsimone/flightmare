#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"

namespace flightlib {

QuadrotorEnv::QuadrotorEnv()
  : QuadrotorEnv(getenv("FLIGHTMARE_PATH") +
                 std::string("/flightlib/configs/quadrotor_env.yaml")) {}

QuadrotorEnv::QuadrotorEnv(const std::string &cfg_path)
  : EnvBase(),
    alive_rew(0.0),
    ori_coeff_(0.0),
    lin_vel_ori_coeff_(0.0),
    lin_vel_mag_coeff_(0.0),
    ang_vel_coeff_(0.0, 0.0, 0.0),
    act_coeff_(0.0, 0.0, 0.0, 0.0),
    goal_state_((Vector<quadenv::kNObs>() << 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0)
                  .finished()) {
  // load configuration file
  base_path = getenv("FLIGHTMARE_PATH");
  // data_path = getenv("FLIGHTMARE_DATASET_PATH");
  cfg_vec_ = YAML::LoadFile(base_path +
                 std::string("/flightlib/configs/vec_env.yaml"));
  rendering = (cfg_vec_["env"]["render"]).as<bool>();
  if(rendering){
    cfg_ = YAML::LoadFile(cfg_path);
  } else {
    cfg_ = YAML::LoadFile(base_path + std::string("/flightlib/configs/quadrotor_env_blind.yaml"));
  }

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  offset_data = (cfg_vec_["data_collection"]["offset_data"]).as<int>();
  collect_data = (cfg_vec_["data_collection"]["collect_data"]).as<bool>();
  subsampling = (cfg_vec_["data_collection"]["subsampling"]).as<int>();
  float depth_scale = (cfg_["quadrotor_env"]["depth_scale"]).as<float>();
  Scalar goal_out_box_prob = (cfg_["quadrotor_env"]["goal_out_box_prob"]).as<Scalar>();
  std::vector<float> camera_ori = (cfg_["quadrotor_env"]["camera_rot"]).as<std::vector<float>>();
  std::vector<float> camera_pos = (cfg_["quadrotor_env"]["camera_pos"]).as<std::vector<float>>();
  frame_height_ = quadenv::frame_height;
  frame_width_ = quadenv::frame_width;
  save_count = offset_data;
  step_count = 0;
  goal_step_count = 0;

  // define input and output dimension for the environment
  obs_dim_ = quadenv::kNObs;
  act_dim_ = quadenv::kNAct;

  // load parameters
  loadParam(cfg_);

  if(rendering) {
    // add camera
    Vector<3> B_r_BC(camera_pos.data());
    // std::cout <<  "camera pos set" << std::endl;
    Matrix<3, 3> R_BC = Quaternion(euler2Quaternion(Vector<3>(camera_ori.data()))).toRotationMatrix();
    Quaternion quat(R_BC);
    // std::cout << "ROTATION MATRIX IS " << std::endl << R_BC << std::endl;
    // logger_.warn("camera pose is: x " + std::to_string(quat. x()) + " y "+ std::to_string(quat.y()) + " z " + std::to_string(quat.z()) + " w " + std::to_string(quat.w()));
    // std::cout <<  "camera ori set" << std::endl;
    rgb_camera = std::make_shared<RGBCamera>();
    rgb_camera->setFOV(quadenv::camera_FOV);
    rgb_camera->setHeight(frame_height_);
    rgb_camera->setWidth(frame_width_);
    rgb_camera->setRelPose(B_r_BC, R_BC);
    rgb_camera->setDepthScale(depth_scale);
    rgb_camera->setPostProcesscing(
                      std::vector<bool>{true, false, false});  // depth, segmentation, optical flow
    quadrotor_ptr_->addRGBCamera(rgb_camera);
  }

  img_ = cv::Mat::zeros(frame_width_, frame_height_, CV_8UC3);
  img_depth_ = cv::Mat::zeros(frame_width_, frame_height_, CV_32FC1);
  img_segm_ = cv::Mat::zeros(frame_width_, frame_height_, CV_8UC3);
  img_mask_ = cv::Mat::zeros(frame_width_, frame_height_, CV_8UC3);

  // define a bounding box
  std::vector<float> world_box = (cfg_["quadrotor_env"]["world_box"]).as<std::vector<float>>();
  world_box_ = Matrix<3, 2>(world_box.data());
  std::vector<float> world_box_evaluation = (cfg_["quadrotor_env"]["world_box_evaluation"]).as<std::vector<float>>();
  world_box_evaluation_ = Matrix<3, 2>(world_box_evaluation.data());
  if(high_level_controller_mode_==1) {
    world_box_ = world_box_evaluation_;
  }
  quadrotor_ptr_->setWorldBox(world_box_);
  goal_range = (world_box_(3) - world_box_(0) - 1)/(2 * sqrt(1 - goal_out_box_prob));
  if(goal_out_box_prob == 0) goal_range *= 0.8;
  // Read goal position
  std::vector<float> goal_relative_position = (cfg_["quadrotor_env"]["goal_relative_position"]).as<std::vector<float>>();
  std::vector<float> omega_max = (cfg_["quadrotor_dynamics"]["omega_max"]).as<std::vector<float>>();
  thrust_scaling = (cfg_["rl"]["thrust_scaling"]).as<Scalar>();
  omega_max_ = Matrix<3, 1>(omega_max.data());
  init_goal_relative_position_ = Matrix<3, 1>(goal_relative_position.data());

  Scalar mass = quadrotor_ptr_->getMass();
  act_mean_ = Vector<quadenv::kNAct>::Ones() * ( - Gz);
  act_std_ = Vector<quadenv::kNAct>::Ones() * (- thrust_scaling * Gz);
  act_mean_(1) = 0;
  act_mean_(2) = 0;
  act_mean_(3) = 0;
  act_std_(1) = omega_max_(0);
  act_std_(2) = omega_max_(1);
  act_std_(3) = omega_max_(2);

  // 6) Survival Reward
  instant_reward_components_(5) = alive_rew - lin_vel_ori_coeff_*LossFunctionSqrt(2.0) + lin_vel_mag_coeff_ - ori_coeff_*LossFunctionSqrt(2.0);
  goal_change_steps = int(norm_dist_(random_gen_) * avg_steps_per_ref / 3 + avg_steps_per_ref);
}

QuadrotorEnv::~QuadrotorEnv() {};


//--------------------------------//
//----      Drone reset       ----//
//--------------------------------//


void QuadrotorEnv::setResetPose(Vector<3> &resetPosition, Vector<3> &resetRotation) {
  resetPosition_ = resetPosition;
  resetRotation_ = resetRotation;
  quadrotor_ptr_->box_center_ = resetPosition_;

  int goalDistance = 35;

  // Adjust goal position accordingly.
  goal_state_(0) = resetPosition(0) + init_goal_relative_position_(0);
  goal_state_(1) = resetPosition(1) + init_goal_relative_position_(1); 
  goal_state_(2) = resetPosition(2) + init_goal_relative_position_(2);

  // Rotation Matrix is equal to rotation of PI/2 around x axis.
  goal_state_(3) = 1;
  goal_state_(4) = 0;
  goal_state_(5) = 0;
  goal_state_(6) = 0;
  goal_state_(7) = 1;
  goal_state_(8) = 0;
  goal_state_(9) = 0;
  goal_state_(10) = 0;
  goal_state_(11) = 1;


  // World box relative to drone reset position.
  world_box_ << world_box_(0)+resetPosition_(0), world_box_(3)+resetPosition_(0), world_box_(1)+resetPosition_(1),
                world_box_(4)+resetPosition_(1), world_box_(2), world_box_(5);

  quadrotor_ptr_->setWorldBox(world_box_);

  // Set quadrotor state.
  setQuadstateToResetState();
}

void QuadrotorEnv::setQuadstateToResetState(){
  quad_state_.setZero();
  quad_act_.setZero();

  quad_state_.x(QS::POSX) = resetPosition_(0);
  quad_state_.x(QS::POSY) = resetPosition_(1);
  quad_state_.x(QS::POSZ) = resetPosition_(2);

  // reset linear velocity
  quad_state_.x(QS::VELX) = init_max_speed * uniform_dist_(random_gen_);
  quad_state_.x(QS::VELY) = init_max_speed * uniform_dist_(random_gen_);
  quad_state_.x(QS::VELZ) = init_max_speed * uniform_dist_(random_gen_);
  // reset orientation
  
  Vector<3> rand_rot_yaw_pitch_roll = Vector<3>(uniform_dist_(random_gen_), uniform_dist_(random_gen_), uniform_dist_(random_gen_))
                                      .cwiseProduct(Vector<3>(init_ori_factor.data())) * 360;
  Vector<4> quat_init_rot = euler2Quaternion(resetRotation_ + rand_rot_yaw_pitch_roll);
  quad_state_.x(QS::ATTW) = quat_init_rot(3);
  quad_state_.x(QS::ATTX) = quat_init_rot(0);
  quad_state_.x(QS::ATTY) = quat_init_rot(1);
  quad_state_.x(QS::ATTZ) = quat_init_rot(2);
  // quad_state_.qx /= quad_state_.qx.norm();
  // std::cout << "DRONE ROTATION MATRIX IS " << std::endl << quad_state_.q().toRotationMatrix() << std::endl;
  // std::cout << "DRONE EULER ANGLES ARE " << std::endl << resetRotation_ << std::endl;
}

bool QuadrotorEnv::reset(Ref<Vector<>> obs, Ref<DepthImage<>> img, const bool random){
  resetObs(obs);

  resetImages(img);

  // resetWorldBox();
  return true;
}

bool QuadrotorEnv::resetObs(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_act_.setZero();

  setQuadstateToResetState();
  
  // reset quadrotor with random states
  quadrotor_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  cmd_.omega.setZero();
  cmd_.collective_thrust = 0;

  // obtain observations
  step_count = 0;
  goal_step_count = 0;

  velocity_ref_inertial_versor = quad_state_.q().toRotationMatrix().col(0);
  ref_velocity_versor_proj = velocity_ref_inertial_versor.segment<2>(0).normalized();
  speed_ref = 0;
  velocity_ref_inertial = velocity_ref_inertial_versor * speed_ref;
  
  if (!high_level_controller_mode_)
  {
    RandomizeReference();
  }

  getObs(obs);

  cumulative_reward_components_ = Vector<7>::Zero();

  return true;
}

//--------------------------------//
//----  Reference generation  ----//
//--------------------------------//

void QuadrotorEnv::CalculateVelocityReference() {
  relativeGoalPosition = (goal_state_.segment<3>(0) - quad_state_.p); 
  goalDistance = relativeGoalPosition.norm();

  if ((!reference_mode_) && (goal_step_count >= goal_change_steps)) {
    RandomizeReference();
  } 
  if (reference_mode_ && ((goal_step_count >= goal_change_steps) || (goalDistance < switch_goal_distance_)) && !permanent_goal) {
    RandomizeReference();
  }
  if (reference_mode_ && (goalDistance < switch_goal_distance_) && permanent_goal) {
    RandomizeReference();
  }

  if (reference_mode_ == 1){
    velocity_ref_inertial_versor = relativeGoalPosition/goalDistance;
    if (speed_ref * VelocityMagnitudeScaling(goalDistance) > speed_close_to_goal) {
      scaled_speed_ref = speed_ref * VelocityMagnitudeScaling(goalDistance);
    }
    velocity_ref_inertial = velocity_ref_inertial_versor * scaled_speed_ref;
  }
  quadrotor_ptr_->reference_velocity_abs_ = velocity_ref_inertial;  

  velocity_ref_body_versor = rot_inertial2body * velocity_ref_inertial_versor;
  velocity_ref_body = velocity_ref_body_versor * speed_ref;
}

void QuadrotorEnv::RandomizeReference() {
  goal_step_count = 0;
  goal_change_steps = int(norm_dist_(random_gen_) * avg_steps_per_ref / 3 + avg_steps_per_ref);

  speed_ref_old = speed_ref;
  speed_ref = maxSpeed * (uniform_dist_(random_gen_) + 1) / 2;

  if(speed_ref < speed_threshold) {
    speed_ref = 0;
    velocity_ref_inertial =  velocity_ref_inertial_versor * speed_ref;
  } else if (uniform_dist_(random_gen_)>0) { // Uniform dist in [-1,+1], so 50% chance
      DrawRandomGoal();
      reference_mode_ = 1;
      if(uniform_dist_(random_gen_)>0.4){ //One goal out of 5 change only if you reach it
        permanent_goal=1;
      }else{
        permanent_goal=0;
      }
  } else {
      DrawRandomVelocity();
      reference_mode_ = 0;
  }
}

void QuadrotorEnv::DrawRandomGoal() {
  Vector<3> goal = Vector<3>(uniform_dist_(random_gen_), uniform_dist_(random_gen_), 0);
  goal_state_.segment<3>(0) = goal * goal_range;
  goal_state_(2) = (uniform_dist_(random_gen_) + 1) / 2 * (world_box_(5) - world_box_(2) - 3) + world_box_(2);
  // resetWorldBox();
  // logger_.warn("Change goal");
  // logger_.warn("New goal: [" + std::to_string(goal_state_(0)) + ", " + std::to_string(goal_state_(1)) + ", " + std::to_string(goal_state_(2)) + " ]");
  // logger_.warn("New speed: " + std::to_string(speed_ref));
  // logger_.warn("New episode lenght: " + std::to_string(goal_change_steps));
  // logger_.warn("New world box is: " + std::to_string(world_box_(0)) + ", " + std::to_string(world_box_(3)) + ", " +
  //                                     std::to_string(world_box_(1)) + ", " + std::to_string(world_box_(4)) + ", " + 
  //                                     std::to_string(world_box_(2)) + ", " + std::to_string(world_box_(5)) + ", ");
}

void QuadrotorEnv::DrawRandomVelocity() {
  // phi = vertical angle from z axiz
  // logger_.warn("Change velocity");
  Scalar phi = std::acos(uniform_dist_(random_gen_));
  // theta = horizontal angle from x axis  
  Scalar theta = 2 * M_PI * std::abs(uniform_dist_(random_gen_));
  // Radius is considered 1
  velocity_ref_inertial_versor << std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi);
  velocity_ref_inertial_versor.normalize();
  velocity_ref_inertial = velocity_ref_inertial_versor * speed_ref;
  // logger_.warn("New episode lenght: " + std::to_string(goal_change_steps));
}

void QuadrotorEnv::setVelocityReference(Ref<Vector<>> reference_velocity_high_level_controller) {
  if (!high_level_controller_mode_) { 
    logger_.warn("Tried to use high level controller without setting the right mode in the configuration.");
    return;
  }
  if (reference_velocity_high_level_controller.size() != 3) {
    logger_.error("reference_velocity_high_level_controller is not a 3 dimensional vector.");
  }
  velocity_ref_inertial = reference_velocity_high_level_controller;
  speed_ref = velocity_ref_inertial.norm();

  quadrotor_ptr_->reference_velocity_abs_ = velocity_ref_inertial;

  velocity_ref_body = rot_inertial2body * velocity_ref_inertial;
  // logger_.warn("reference_velocity_ : " + std::to_string(velocity_ref_inertial(0))+ " " + std::to_string(velocity_ref_inertial(1))+ " " + std::to_string(velocity_ref_inertial(2)));
  // logger_.warn("velocity_ref_inertial_versor : " + std::to_string(velocity_ref_inertial_versor(0))+ " " + std::to_string(velocity_ref_inertial_versor(1))+ " " + std::to_string(velocity_ref_inertial_versor(2)));
  // logger_.warn("reference_velocity_magnitude_ : " + std::to_string(speed_ref));
}

bool QuadrotorEnv::resetImages(Ref<DepthImage<>> img) {  
  img_depth_ = cv::Mat::zeros(frame_width_, frame_height_, CV_32FC1);

  getImages(img);
    
  terminal_reason_ = 0;
  terminal_reward_ = 0;

  return true;
}

//--------------------------------//
//----      Observations      ----//
//--------------------------------//

bool QuadrotorEnv::getImages(Ref<DepthImage<>> img) {
  
  //Update image observation

  if(rendering && !collect_data){
    rgb_camera->getDepthMap(img_depth_);

    cv::cv2eigen(img_depth_, depth_img_mat_);
    Map<DepthImage<>> img_vec_(depth_img_mat_.data(), depth_img_mat_.size());
    img.block<quadenv::frame_height*quadenv::frame_width,1>(0,0) = img_vec_;
  }

  return true;
}

bool QuadrotorEnv::getObs(Ref<Vector<>> obs) {

  quadrotor_ptr_->getState(&quad_state_);
  
  rot_matrix = quad_state_.q().toRotationMatrix();
  rot_inertial2body = quad_state_.q().inverse();
  
   if (!high_level_controller_mode_) {
    CalculateVelocityReference();
  }

  // Give rotation matrix one column at a time. So keep it's vectors consistent.
  quad_obs_ << quad_state_.p, rot_matrix.col(0), rot_matrix.col(1), rot_matrix.col(2),
               rot_inertial2body * quad_state_.v, quad_state_.w;

  // logger_.warn("Drone position is: " + std::to_string(quad_state_.p(0) - goal_state_(0)));
  obs.segment<quadenv::kNObs>(quadenv::kObs) = quad_obs_;
  
  return true;
}


//--------------------------------//
//----          Step          ----//
//--------------------------------//


Scalar QuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  cmd_.collective_thrust = quad_act_(0);
  cmd_.omega = quad_act_.tail<3>();

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // update observations
  step_count++;
  goal_step_count++;
  getObs(obs);

  Matrix<3, 3> rot = quad_state_.q().toRotationMatrix();
  Vector<3> drone_velocity_body_frame = quad_obs_.segment<quadenv::kNLinVel>(quadenv::kLinVel); 
  Scalar velocity_norm = drone_velocity_body_frame.norm();
  Scalar goal_plane_proj = relativeGoalPosition.segment<2>(0).norm();
  // Vector<2> drone_front_axis_proj = rot.block<2,1>(0,0).normalized();

  // Fading coefficient used to smooth the transitions between two different references
  // the idea is that when the reference is changed the agent should not pay the error 
  // immediatly but gradually 
  Scalar fading_coeff =  pow(std::min(float(1), float(goal_step_count)/float(vel_fading_sat)), vel_fading_exp);

  Vector<3> velocity_ref_mask = Vector<3>::Ones();
  Scalar min_distance_to_wall = 1;

  if (quad_state_.x(QS::POSX) <= world_box_(0) + min_distance_to_wall & velocity_ref_inertial(0) < 0) {
    velocity_ref_mask(0) = 0;
    // logger_.warn("out of bound x " + std::to_string(quad_state_.x(QS::POSX)) + " outside " + "[" + std::to_string(world_box_(0) + 0.5) + ", " + std::to_string(world_box_(3) - 0.5) + "]");
  }
  if (quad_state_.x(QS::POSX) >= world_box_(3) - min_distance_to_wall & velocity_ref_inertial(0) > 0) {
    velocity_ref_mask(0) = 0;
    // logger_.warn("out of bound x " + std::to_string(quad_state_.x(QS::POSX)) + " outside " + "[" + std::to_string(world_box_(0) + 0.5) + ", " + std::to_string(world_box_(3) - 0.5) + "]");
  }
  // Check out of bounds y.
  if (quad_state_.x(QS::POSY) <= world_box_(1) + min_distance_to_wall & velocity_ref_inertial(1) < 0) {
    velocity_ref_mask(1) = 0;
    // logger_.warn("out of bound y " + std::to_string(quad_state_.x(QS::POSY)) + " outside " + "[" + std::to_string(world_box_(1) + 0.5) + ", " + std::to_string(world_box_(4) - 0.5) + "]");
  }
  // Check out of bounds y.
  if (quad_state_.x(QS::POSY) >= world_box_(4) - min_distance_to_wall & velocity_ref_inertial(1) > 0) {
    velocity_ref_mask(1) = 0;
    // logger_.warn("out of bound y " + std::to_string(quad_state_.x(QS::POSY)) + " outside " + "[" + std::to_string(world_box_(1) + 0.5) + ", " + std::to_string(world_box_(4) - 0.5) + "]");
  }
  // Check out of bounds z.
  if (quad_state_.x(QS::POSZ) <= world_box_(2) + min_distance_to_wall  & velocity_ref_inertial(2) < 0) {
    velocity_ref_mask(2) = 0;
    // logger_.warn("out of bound z " + std::to_string(quad_state_.x(QS::POSZ)) + " outside " + "[" + std::to_string(world_box_(2) + 0.5) + ", " + std::to_string(world_box_(5) - 0.5) + "]");
  }
  // Check out of bounds z.
  if (quad_state_.x(QS::POSZ) >= world_box_(5) - min_distance_to_wall & velocity_ref_inertial(2) > 0) {
    velocity_ref_mask(2) = 0;
    // logger_.warn("out of bound z " + std::to_string(quad_state_.x(QS::POSZ)) + " outside " + "[" + std::to_string(world_box_(2) + 0.5) + ", " + std::to_string(world_box_(5) - 0.5) + "]");
  }

  // --------- Reward function design --------- //
  // 1) lin_vel_ori_reward
  // 2) lin_vel_mag_reward
  // 3) ori_reward
  // 4) ang_vel_reward
  // 5) act_reward
  // 6) survival_reward

  if ( speed_ref > speed_threshold) { // Velocity above which we want to move. 
    // Reference velocity tracking.
    // 1) Reference velocity orientation tracking.
    if(velocity_ref_inertial.segment<2>(0).norm() > 0.1){
      ref_velocity_versor_proj = velocity_ref_inertial.cwiseProduct(velocity_ref_mask).segment<2>(0).normalized();
      drone_front_axis_proj = rot.block<2,1>(0,0).normalized();
    }

    
    vel_orientation_factor = velocity_ref_inertial.cwiseProduct(velocity_ref_mask).normalized().dot(quad_state_.v / velocity_norm);
    Scalar velocity_orientation_err = velocity_norm > speed_threshold ?
                  1 - vel_orientation_factor
                  : 1 - drone_front_axis_proj.dot(ref_velocity_versor_proj);
    instant_reward_components_(0) = lin_vel_ori_coeff_ * LossFunctionSqrt(velocity_orientation_err) * fading_coeff;
    // logger_.warn("velocity_orientation_err: " + std::to_string(velocity_orientation_err));
    
    // 2) Reference velocity magnitude tracking. 
    // If this doesn't work try using projected velocity.
    // Scalar lin_vel_mag_reward;
    Scalar smooth_speed_ref = fading_coeff * speed_ref + (1-fading_coeff)*speed_ref_old;
    Scalar velocity_magnitude_err = (velocity_ref_inertial - quad_state_.v).cwiseProduct(velocity_ref_mask).norm()/(speed_ref);
    instant_reward_components_(1) = lin_vel_mag_coeff_ * LossFunctionAtan(velocity_magnitude_err)  * fading_coeff;
    // logger_.warn("velocity_magnitude_err : " + std::to_string(velocity_magnitude_err));

  } else { // If velocity too small just stay still.
    // Reference velocity tracking.
    // 1) Reference velocity orientation tracking.
    
    instant_reward_components_(0) = 0;
    
    // 2) Reference velocity magnitude tracking. 
    instant_reward_components_(1) = lin_vel_still_coeff_ * LossFunctionAtan(velocity_norm) * fading_coeff;
  }

  // 3) Orientation such as goal is in FOV, drone should be directed in the direction of the velocity.
  // Scalar ori_reward;

  Scalar orientation_err = velocity_ref_inertial.segment<2>(0).norm() > 0.1 ? 
                          1 - drone_front_axis_proj.dot(ref_velocity_versor_proj)
                          : 0 ;
  instant_reward_components_(2) = ori_coeff_ * (LossFunctionSqrt(orientation_err));
  // logger_.warn("orientation_err: " + std::to_string(instant_reward_components_(2)));

  // 4) angular velocity tracking
  // Scalar ang_vel_reward;
  Scalar angle_ref_x = atan2(ref_velocity_versor_proj(0) * drone_front_axis_proj(1) - drone_front_axis_proj(0) * ref_velocity_versor_proj(1),
                             ref_velocity_versor_proj.dot(drone_front_axis_proj));
  Vector<3> omega_offset = velocity_ref_inertial.segment<2>(0).norm() > 0.1 ?
                          Vector<3>(0, 0, -kp_ang_velz * angle_ref_x)
                          : Vector<3>(0, 0, 0);
  // logger_.warn("Angle between reference and x axis is " + std::to_string(angle_ref_x * 180 / M_PI));
  instant_reward_components_(3) = - ang_vel_coeff_.cwiseProduct(quad_obs_.segment<quadenv::kNAngVel>(quadenv::kAngVel) - omega_offset).squaredNorm();

  // 5) control action penalty
  // Scalar act_reward;
  //  This offset is meant to penalize thrusts that are not exerting g along the vertical axis not the drone axis up to 30 deg
  Scalar thrust_offset = rot(2, 2) > 0.866 ?
                        (1/rot(2,2) - 1) / thrust_scaling
                        :(1.1547 - 1) / thrust_scaling;
  instant_reward_components_(4) = - (act - Vector<4>(thrust_offset, 0, 0, omega_offset(2) / omega_max_(2))).cwiseProduct(act_coeff_).norm();


  if(verbose_level_ == 1) {
    LogRewardInfo();
  }

  return instant_reward_components_.sum();
}

Scalar QuadrotorEnv::stepUnity(Ref<DepthImage<>> img) {
  
  // Update images
  getImages(img);
  instant_reward_components_(6) = 0;
  // All rewards relative to unity
  Scalar unity_rew = instant_reward_components_(6);

  return unity_rew;
}

//--------------------------------//
//----    Terminal analysis   ----//
//--------------------------------//



bool QuadrotorEnv::isTerminalState(Scalar &reward) {

  if(goal_step_count > vel_off_steps && ((quad_obs_.segment<quadenv::kNVelRef>(quadenv::kVelRef).norm() > vel_off_mag) || (vel_orientation_factor < vel_off_ori))) {
    terminal_reason_ = quadenv::velocityTooOff;
    reward = velocity_too_off_penalty_;
    terminal_reward_ = reward;
    // logger_.error("Velocity off by more than 5 m/s");
    // logger_.error("Velocity error is " + std::to_string(quad_obs_.segment<quadenv::kNVelRef>(quadenv::kVelRef).norm()));
    return true;
  }

  if(step_count > max_steps) {
    terminal_reason_ = quadenv::termTimeIsUp;
    return true;
  }

  // Check out of bounds x. 
  if (quad_state_.x(QS::POSX) <= world_box_(0) + 0.5 || quad_state_.x(QS::POSX) >= world_box_(3) - 0.5 ) {
    reward = world_box_penalty_ + DistanceToGoalPenalty() + OnCollisionVelocityPenalty();
    terminal_reason_ = quadenv::termCrashInWorldBox;
    // logger_.warn("out of bound x " + std::to_string(quad_state_.x(QS::POSX)) + " outside " + "[" + std::to_string(world_box_(0) + 0.5) + ", " + std::to_string(world_box_(3) - 0.5) + "]");
  }
  // Check out of bounds y.
  if (quad_state_.x(QS::POSY) <= world_box_(1) + 0.5 || quad_state_.x(QS::POSY) >= world_box_(4) - 0.5 ) {
    reward = world_box_penalty_ + DistanceToGoalPenalty() + OnCollisionVelocityPenalty();
    terminal_reason_ = quadenv::termCrashInWorldBox;
    // logger_.warn("out of bound y " + std::to_string(quad_state_.x(QS::POSY)) + " outside " + "[" + std::to_string(world_box_(1) + 0.5) + ", " + std::to_string(world_box_(4) - 0.5) + "]");
  }
  // Check out of bounds z.
  if (quad_state_.x(QS::POSZ) <= world_box_(2) + 0.5  || quad_state_.x(QS::POSZ) >= world_box_(5) - 0.5 ) {
    reward = world_box_penalty_ + DistanceToGoalPenalty() + OnCollisionVelocityPenalty();
    terminal_reason_ = quadenv::termCrashInWorldBox;
    // logger_.warn("out of bound z " + std::to_string(quad_state_.x(QS::POSZ)) + " outside " + "[" + std::to_string(world_box_(2) + 0.5) + ", " + std::to_string(world_box_(5) - 0.5) + "]");
  }

  if (reward==0.0) {
    return false;
  }

  terminal_reward_ = reward;
  return true;
}

Scalar QuadrotorEnv::OnCollisionVelocityPenalty() {
  return (quad_obs_.segment<quadenv::kNLinVel>(quadenv::kLinVel)).norm()*crash_vel_coeff_;
}

bool QuadrotorEnv::isTerminalStateUnity(Scalar &reward) {
  // Check collision. 
  // Need justHadCollision flag because otherwise collision with a single object is thrown for two consecutive timesteps.
  if (quadrotor_ptr_->getCollision() && step_count>4) {
    // Collision is actually happening. 
    if (!justHadCollision)
    {
      reward = crash_penalty_;
      reward += DistanceToGoalPenalty();
      reward += OnCollisionVelocityPenalty();
      // logger_.warn("quadrotor_ptr_->getCollision : " + std::to_string(quadrotor_ptr_->getCollision()));
      justHadCollision = true;
      terminal_reward_ = reward;
      terminal_reason_ = quadenv::termCollisionWithObstacle;
      return true;
    } else { // Collision is due to previous timestep.
      collision_step_count++;
      if(collision_step_count == 10){
      justHadCollision = false;
      collision_step_count = 0;
      }
    }
  }

  reward = 0.0;
  return false;
}

Scalar QuadrotorEnv::DistanceToGoalPenalty() {
  return goalDistance*crash_dist_coeff_;
}

void QuadrotorEnv::resetWorldBox() {
  Scalar x_lower = std::min(goal_state_(0) - 5, quad_state_.x(QS::POSX) - 5);
  Scalar x_upper = std::max(goal_state_(0) + 5, quad_state_.x(QS::POSX) + 5);
  Scalar y_lower = std::min(goal_state_(1) - 5, quad_state_.x(QS::POSY) - 5);
  Scalar y_upper = std::max(goal_state_(1) + 5, quad_state_.x(QS::POSY) + 5);
  Scalar z_lower = std::min(goal_state_(2) - 4, quad_state_.x(QS::POSZ) - 4);
  Scalar z_upper = std::max(goal_state_(2) + 4, quad_state_.x(QS::POSZ) + 4);
  world_box_ << x_lower, x_upper, y_lower, y_upper, z_lower, z_upper;
  quadrotor_ptr_->setWorldBox(world_box_);
}

void QuadrotorEnv::updateExtraInfo(){
  // set additional information about environment to analyze learning progress.
  if (high_level_controller_mode_) {
    extra_info_[relative_pos_x_key_] = quad_state_.p(0) - resetPosition_(0);
    extra_info_[relative_pos_y_key_] = quad_state_.p(1) - resetPosition_(1);
    extra_info_[relative_pos_z_key_] = quad_state_.p(2);
  }
  extra_info_[lin_vel_ori_rew_key_] = (float)instant_reward_components_(0);
  extra_info_[lin_vel_mag_rew_key_] = (float)instant_reward_components_(1);
  extra_info_[ori_rew_key_]         = (float)instant_reward_components_(2);
  extra_info_[ang_vel_rew_key_]     = (float)instant_reward_components_(3);
  extra_info_[act_rew_key_]         = (float)instant_reward_components_(4);
  extra_info_[survival_rew_key_]    = (float)instant_reward_components_(5);
  extra_info_[term_rew_key_]        = (float)terminal_reward_;
  extra_info_[term_reason_key_]     = (float)terminal_reason_;
}

void QuadrotorEnv::LogRewardInfo() {
  cumulative_reward_components_(0) += instant_reward_components_(0);
  cumulative_reward_components_(1) += instant_reward_components_(1);
  cumulative_reward_components_(2) += instant_reward_components_(2);
  cumulative_reward_components_(3) += instant_reward_components_(3);
  cumulative_reward_components_(4) += instant_reward_components_(4);
  cumulative_reward_components_(5) += instant_reward_components_(5);
  cumulative_reward_components_(6) += instant_reward_components_(6);

  logger_.warn("lin_vel_ori_reward: " + std::to_string(instant_reward_components_(0)));
  logger_.warn("lin_vel_mag_reward: " + std::to_string(instant_reward_components_(1)));
  logger_.warn("ori_reward: " + std::to_string(instant_reward_components_(2)));
  logger_.warn("ang_vel_reward: " + std::to_string(instant_reward_components_(3)));
  logger_.warn("act_reward: " + std::to_string(instant_reward_components_(4)));
  logger_.warn("survival_reward_: " + std::to_string(instant_reward_components_(5)));
  logger_.warn("---");
  logger_.warn("lin_vel_ori_reward cumulative: " + std::to_string(cumulative_reward_components_(0)));
  logger_.warn("lin_vel_mag_reward cumulative: " + std::to_string(cumulative_reward_components_(1)));
  logger_.warn("ori_reward cumulative: " + std::to_string(cumulative_reward_components_(2)));
  logger_.warn("ang_vel_reward cumulative: " + std::to_string(cumulative_reward_components_(3)));
  logger_.warn("act_reward cumulative: " + std::to_string(cumulative_reward_components_(4)));
  logger_.warn("survival_reward_ cumulative: " + std::to_string(cumulative_reward_components_(5)));
}



bool QuadrotorEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
    maxSpeed = cfg["quadrotor_env"]["max_speed"].as<Scalar>();
    speed_threshold = cfg["quadrotor_env"]["speed_threshold"].as<Scalar>();
    speed_close_to_goal = speed_threshold + 0.5;
    init_max_speed = cfg["quadrotor_env"]["init_max_speed"].as<Scalar>() / sqrt(3);
    init_ori_factor = cfg["quadrotor_env"]["init_ori_factor"].as<std::vector<float>>();
  } else {
    return false;
  }

  if (cfg["rl"]) {
    // load reinforcement learning related parameters
    max_steps = (cfg_["rl"]["max_episode_steps"]).as<int>();
    avg_steps_per_ref = (cfg_["rl"]["avg_steps_per_ref"]).as<int>();
    switch_goal_distance_ = (cfg_["velocity_planner"]["goal_change_dist"]).as<Scalar>();
    vel_off_mag = (cfg_["rl"]["vel_off_mag"]).as<Scalar>();
    vel_fading_sat = (cfg_["rl"]["vel_fading_saturation"]).as<int>();
    vel_fading_exp = (cfg_["rl"]["vel_fading_exp"]).as<int>();
    vel_off_ori = (cfg_["rl"]["vel_off_ori"]).as<Scalar>();
    vel_off_steps = (cfg_["rl"]["vel_off_steps"]).as<int>();
    alive_rew = cfg["rl"]["alive_rew"].as<Scalar>();
    ori_coeff_ = cfg["rl"]["ori_coeff"].as<Scalar>();
    lin_vel_ori_coeff_ = cfg["rl"]["lin_vel_ori_coeff"].as<Scalar>();
    lin_vel_mag_coeff_ = cfg["rl"]["lin_vel_mag_coeff"].as<Scalar>();
    lin_vel_still_coeff_ = cfg["rl"]["lin_vel_still_coeff"].as<Scalar>();
    std::vector<float> ang_vel_coeff = cfg["rl"]["ang_vel_coeff"].as<std::vector<float>>();
    ang_vel_coeff_ << ang_vel_coeff[0], ang_vel_coeff[0], ang_vel_coeff[1]; 
    std::vector<float> act_coeff = (cfg["rl"]["act_coeff"]).as<std::vector<float>>();
    act_coeff_ << act_coeff[0], act_coeff[1], act_coeff[1], act_coeff[2]; 
    crash_vel_coeff_ = cfg["rl"]["crash_vel_coeff"].as<Scalar>();
    crash_dist_coeff_ = cfg["rl"]["crash_dist_coeff"].as<Scalar>();
    crash_dist_coeff_ = cfg["rl"]["crash_dist_coeff"].as<Scalar>();
    crash_penalty_ = cfg["rl"]["crash_penalty"].as<Scalar>();
    world_box_penalty_ = cfg["rl"]["world_box_penalty"].as<Scalar>();
    velocity_too_off_penalty_ = cfg["rl"]["velocity_too_off_penalty"].as<Scalar>();
    kp_ang_velz = cfg["rl"]["kp_ang_velz"].as<Scalar>();
  } else {
    return false;
  }

  if (cfg["velocity_planner"]) {
    // load reinforcement learning related parameters
    high_level_controller_mode_ = (cfg["velocity_planner"]["high_level_controller_mode"]).as<bool>();
    switch_goal_distance_ = (cfg["velocity_planner"]["goal_change_dist"]).as<Scalar>();
  } else {
    return false;
  }

  return true;
}

bool QuadrotorEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool QuadrotorEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void QuadrotorEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const QuadrotorEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib