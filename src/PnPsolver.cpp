// created by liu han on 2024/1/27
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
// Modified by removing ROS dependencies
// Submodule of HeliosRobotSystem
#include "PnPSolver.hpp"

namespace helios_cv {

PnPSolver::PnPSolver(
    const std::array<double, 9>& camera_matrix,
    const std::vector<double>& dist_coeffs
):
    camera_matrix_(cv::Mat(3, 3, CV_64F, const_cast<double*>(camera_matrix.data())).clone()),
    dist_coeffs_(cv::Mat(1, 5, CV_64F, const_cast<double*>(dist_coeffs.data())).clone()) {
    // Unit: m
    double half_y {};
    double half_z {};

    // Start from bottom left in clockwise order
    // Model coordinate: x forward, y left, z up
    // Small armor
    // Unit: m
    half_y = small_armor_width / 2.0 / 1000.0;
    half_z = small_armor_height / 2.0 / 1000.0;

    small_armor_points_.emplace_back(cv::Point3f(0, half_y, -half_z));
    small_armor_points_.emplace_back(cv::Point3f(0, half_y, half_z));
    small_armor_points_.emplace_back(cv::Point3f(0, -half_y, half_z));
    small_armor_points_.emplace_back(cv::Point3f(0, -half_y, -half_z));

    // Large armor
    half_y = large_armor_width / 2.0 / 1000.0;
    half_z = large_armor_height / 2.0 / 1000.0;

    large_armor_points_.emplace_back(cv::Point3f(0, half_y, -half_z));
    large_armor_points_.emplace_back(cv::Point3f(0, half_y, half_z));
    large_armor_points_.emplace_back(cv::Point3f(0, -half_y, half_z));
    large_armor_points_.emplace_back(cv::Point3f(0, -half_y, -half_z));

    // Unit: m
    half_y = energy_armor_width_ / 2.0 / 1000.0;
    half_z = energy_armor_height / 2.0 / 1000.0;
    double r_poition = 0.692;

    // Start from bottom left in clockwise order, R last
    // Model coordinate: x forward, y left, z up
    // Unit: m
    energy_armor_points_.emplace_back(cv::Point3f(0, 0, -half_z));
    energy_armor_points_.emplace_back(cv::Point3f(0, half_y, 0));
    energy_armor_points_.emplace_back(cv::Point3f(0, 0, half_z));
    energy_armor_points_.emplace_back(cv::Point3f(0, -half_y, 0));
    // R center
    energy_armor_points_.emplace_back(cv::Point3f(0, 0, -r_poition));
}

bool PnPSolver::solve_pnp(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec) {
    image_armor_points_.clear();
    // Fill in image points
    image_armor_points_.emplace_back(armor.left_light.bottom);
    image_armor_points_.emplace_back(armor.left_light.top);
    image_armor_points_.emplace_back(armor.right_light.top);
    image_armor_points_.emplace_back(armor.right_light.bottom);

    // Solve pnp
    if (armor.type == ArmorType::SMALL) {
        object_points_ = small_armor_points_;
    } else if (armor.type == ArmorType::LARGE) {
        object_points_ = large_armor_points_;
    } else if (armor.type == ArmorType::ENERGY_TARGET || armor.type == ArmorType::ENERGY_FAN) {
        image_armor_points_.emplace_back(armor.center);
        object_points_ = energy_armor_points_;
    } else {
        return false;
    }

    return cv::solvePnP(
        object_points_,
        image_armor_points_,
        camera_matrix_,
        dist_coeffs_,
        rvec,
        tvec,
        false,
        cv::SOLVEPNP_IPPE
    );
}

bool PnPSolver::solve_pose(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec) {
    if (solve_pnp(armor, rvec, tvec)) {
        return true;
    } else {
        return false;
    }
}

float PnPSolver::calculateDistanceToCenter(const cv::Point2f& image_point) {
    float cx = camera_matrix_.at<double>(0, 2);
    float cy = camera_matrix_.at<double>(1, 2);
    return cv::norm(image_point - cv::Point2f(cx, cy));
}

ArmorProjectYaw::ArmorProjectYaw(
    const std::array<double, 9>& camera_matrix,
    const std::vector<double>& dist_coeffs
):
    PnPSolver(camera_matrix, dist_coeffs) {
    pthis_ = this;
}

template<typename T>
bool ArmorProjectYaw::CostFunctor::operator()(const T* const yaw, T* residual) const {
    // Caculate pose in imu
    Eigen::Matrix<T, 3, 3> rotation_mat, R_x, R_y, R_z;
    R_x << T(1), T(0), T(0), T(0), ceres::cos(T(pthis_->roll_)), -ceres::sin(T(pthis_->roll_)),
        T(0), ceres::sin(T(pthis_->roll_)), ceres::cos(T(pthis_->roll_));
    R_y << ceres::cos(T(pthis_->pitch_)), T(0), ceres::sin(T(pthis_->pitch_)), T(0), T(1), T(0),
        -ceres::sin(T(pthis_->pitch_)), T(0), ceres::cos(T(pthis_->pitch_));
    R_z << ceres::cos(*yaw), -ceres::sin(*yaw), T(0), ceres::sin(*yaw), ceres::cos(*yaw), T(0),
        T(0), T(0), T(1);
    rotation_mat = R_z * R_y * R_x;
    // Convert pose to camera
    Eigen::Matrix<T, 3, 3> odom2cam_r;
    odom2cam_r << T(pthis_->odom2cam_r_(0, 0)), T(pthis_->odom2cam_r_(0, 1)),
        T(pthis_->odom2cam_r_(0, 2)), T(pthis_->odom2cam_r_(1, 0)), T(pthis_->odom2cam_r_(1, 1)),
        T(pthis_->odom2cam_r_(1, 2)), T(pthis_->odom2cam_r_(2, 0)), T(pthis_->odom2cam_r_(2, 1)),
        T(pthis_->odom2cam_r_(2, 2));
    rotation_mat = odom2cam_r * rotation_mat;
    // Project points
    Eigen::Matrix<T, 3, 1> tvec;
    tvec << T(pthis_->tvec_.at<double>(0, 0)), T(pthis_->tvec_.at<double>(1, 0)),
        T(pthis_->tvec_.at<double>(2, 0));
    std::vector<Eigen::Vector<T, 3>> points;
    for (const auto& point: pthis_->object_points_) {
        Eigen::Matrix<T, 3, 1> point_eigen;
        point_eigen << T(point.x), T(point.y), T(point.z);
        points.emplace_back(rotation_mat * point_eigen + tvec);
    }
    Eigen::Matrix<T, 3, 3> camera_matrix;
    camera_matrix << T(pthis_->camera_matrix_.at<double>(0, 0)),
        T(pthis_->camera_matrix_.at<double>(0, 1)), T(pthis_->camera_matrix_.at<double>(0, 2)),
        T(pthis_->camera_matrix_.at<double>(1, 0)), T(pthis_->camera_matrix_.at<double>(1, 1)),
        T(pthis_->camera_matrix_.at<double>(1, 2)), T(pthis_->camera_matrix_.at<double>(2, 0)),
        T(pthis_->camera_matrix_.at<double>(2, 1)), T(pthis_->camera_matrix_.at<double>(2, 2));
    std::vector<Eigen::Vector<T, 3>> projected_points;
    for (const auto& point: points) {
        Eigen::Vector<T, 3> projected_point;
        projected_point = camera_matrix / point[2] * point;
        projected_points.emplace_back(projected_point);
    }
    // Caculate the difference between projected points and image points (Use Vector's angle)
    auto get_angle = [](const Eigen::Vector<T, 3>& v1, const Eigen::Vector<T, 3>& v2) {
        return ceres::acos(v1.dot(v2) / (v1.norm() * v2.norm()));
    };
    Eigen::Vector<T, 3> image_vector, projected_vector;
    residual[0] = T(0);
    for (int i = 0; i < 4; i++) {
        image_vector << T(
            pthis_->image_armor_points_[(i + 1) % 4].x - pthis_->image_armor_points_[i].x
        ),
            T(pthis_->image_armor_points_[(i + 1) % 4].y - pthis_->image_armor_points_[i].y), T(1);
        projected_vector << projected_points[(i + 1) % 4][0] - projected_points[i][0],
            projected_points[(i + 1) % 4][1] - projected_points[i][1], T(1);
        residual[0] += get_angle(image_vector, projected_vector);
    }
    return true;
}

void ArmorProjectYaw::update_transform_info(BaseTransformInfo* transform_info) {
    auto transform_infos = dynamic_cast<ArmorTransformInfo*>(transform_info);
    odom2cam_r_ = transform_infos->odom2cam_r.toRotMat3x3();
    cam2odom_r_ = transform_infos->cam2odom_r.toRotMat3x3();
    is_transform_info_updated_ = true;
}

double ArmorProjectYaw::diff_function(double yaw) {
    // Caculate rotation matrix
    cv::Mat rotation_matrix;
    get_rotation_matrix(yaw, rotation_matrix);
    rotation_matrix = odom2cam_r_ * rotation_matrix;
    // Turn rotation matrix into rotation vector
    // caculate projection points
    cv::projectPoints(
        object_points_,
        rotation_matrix,
        tvec_,
        camera_matrix_,
        dist_coeffs_,
        projected_points_
    );
    // Caculate the difference between projected points and image points by angle between vectors
    double diff = 0;
    Eigen::Vector3d image_vector[2], projected_vector[2];
    image_vector[0] << image_armor_points_[3].x - image_armor_points_[0].x,
        image_armor_points_[3].y - image_armor_points_[0].y, 1;
    image_vector[1] << image_armor_points_[1].x - image_armor_points_[2].x,
        image_armor_points_[1].y - image_armor_points_[2].y, 1;
    projected_vector[0] << projected_points_[3].x - projected_points_[0].x,
        projected_points_[3].y - projected_points_[0].y, 1;
    projected_vector[1] << projected_points_[1].x - projected_points_[0].x,
        projected_points_[1].y - projected_points_[0].y, 1;
    Eigen::Vector3d image_cross = image_vector[0].cross(image_vector[1]);
    Eigen::Vector3d projected_cross = projected_vector[0].cross(projected_vector[1]);
    double raw_angle =
        image_vector[0].dot(image_vector[1]) / (image_vector[0].norm() * image_vector[1].norm());
    double project_angle;
    if (image_cross.dot(projected_cross) >= 0) {
        project_angle = projected_vector[0].dot(projected_vector[1])
            / (projected_vector[0].norm() * projected_vector[1].norm());
    } else {
        project_angle = -projected_vector[0].dot(projected_vector[1])
            / (projected_vector[0].norm() * projected_vector[1].norm());
    }
    return raw_angle - project_angle;
}

void ArmorProjectYaw::draw_projection_points(cv::Mat& image) {
    if (projected_points_.empty()) {
        logger_.debug("empty projection");
        return;
    }
    for (int i = 0; i < 4; i++) {
        cv::circle(image, projected_points_[i], i + 1, cv::Scalar(0, 0, 255), -1);
        // cv::circle(image, image_points_[i], i + 1, cv::Scalar(255, 0, 0), 2);
    }
    cv::line(image, projected_points_[0], projected_points_[2], cv::Scalar(255, 255, 0), 2);
    cv::line(image, projected_points_[1], projected_points_[3], cv::Scalar(255, 255, 0), 2);
    cv::line(image, projected_points_[0], projected_points_[1], cv::Scalar(255, 255, 0), 2);
    cv::line(image, projected_points_[1], projected_points_[2], cv::Scalar(255, 255, 0), 2);
    cv::line(image, projected_points_[2], projected_points_[3], cv::Scalar(255, 255, 0), 2);
    cv::line(image, projected_points_[3], projected_points_[0], cv::Scalar(255, 255, 0), 2);
    projected_points_.clear();
}

double ArmorProjectYaw::phi_optimization(double left, double right, double eps) {
    // Make fibonacci sequence
    std::vector<double> fn(2);
    std::vector<double> fib_seq;
    fn[0] = 1, fn[1] = 1;
    double f_lambda = 0, f_mu = 0;
    int fib_cnt;
    for (fib_cnt = 2; fn[fib_cnt - 1] < 1.0 / eps; fib_cnt++) {
        fn.push_back(fn[fib_cnt - 2] + fn[fib_cnt - 1]);
    }
    fib_cnt--;
    double lambda = left + (fn[fib_cnt - 2] / fn[fib_cnt]) * (right - left);
    double mu = left + (fn[fib_cnt - 1] / fn[fib_cnt]) * (right - left);
    f_lambda = diff_function(lambda);
    f_mu = diff_function(mu);
    for (int i = 0; i < fib_cnt - 2; i++) {
        if (f_lambda < f_mu) {
            right = mu;
            mu = lambda;
            f_mu = f_lambda;
            lambda = left + (fn[fib_cnt - i - 3] / fn[fib_cnt - i - 1]) * (right - left);
            f_lambda = diff_function(lambda);
        } else {
            left = lambda;
            lambda = mu;
            f_lambda = f_mu;
            mu = left + (fn[fib_cnt - i - 2] / fn[fib_cnt - i - 1]) * (right - left);
            f_mu = diff_function(mu);
        }
    }
    f_mu = diff_function(mu);
    if (f_lambda < f_mu) {
        right = mu;
    } else {
        left = lambda;
    }
    return 0.5 * (left + right);
}

void ArmorProjectYaw::get_rotation_matrix(double yaw, cv::Mat& rotation_mat) const {
    // caculate rotation matrix
    // for more see paper: https://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
    // Calculate rotation about x axis
    // clang-format off
  cv::Mat R_x = (cv::Mat_<double>(3,3) <<
              1,       0,               0,
              0,       std::cos(roll_), -std::sin(roll_),
              0,       std::sin(roll_), std::cos(roll_));
    
  // Calculate rotation about y axis
  cv::Mat R_y = (cv::Mat_<double>(3,3) <<
              std::cos(pitch_),    0,      std::sin(pitch_),
              0,                   1,      0,
              -std::sin(pitch_),   0,      std::cos(pitch_));
    
  // Calculate rotation about z axis
  cv::Mat R_z = (cv::Mat_<double>(3,3) <<
              std::cos(yaw),    -std::sin(yaw),      0,
              std::sin(yaw),    std::cos(yaw),       0,
              0,                0,                   1);
    // clang-format on
    rotation_mat = R_z * R_y * R_x;
}

bool ArmorProjectYaw::solve_pose(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec) {
    /*Solve PnP First To Get Position Info*/
    // Fill in image points
    if (!use_projection_) {
        return solve_pnp(armor, rvec, tvec);
    } else {
        if (!solve_pnp(armor, rvec, tvec)) {
            return false;
        }
    }
    if (!is_transform_info_updated_) {
        logger_.warn("PnP Solve done, but transform info not updated, skipping");
        return true;
    }
    /*Start Reprojection*/
    double yaw = 0;
    tvec_ = tvec;
    armor_angle_ = armor.angle;
    // Choose pitch value
    if (armor.number == "outpost") {
        pitch_ = -15.0 * M_PI / 180.0; // Convert to radians
    } else {
        pitch_ = 15.0 * M_PI / 180.0; // Convert to radians
    }
    // Get min diff yaw
    ceres::Problem problem;
    ceres::CostFunction* costfunctor =
        new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(costfunctor, nullptr, &yaw);
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // Get draw points
    diff_function(yaw);
    // Caculate rotation matrix
    get_rotation_matrix(yaw, rvec);
    rvec = odom2cam_r_ * rvec;
    return true;
}

void ArmorProjectYaw::set_use_projection(bool use_projection) {
    use_projection_ = use_projection;
}

bool ArmorProjectYaw::use_projection() {
    return use_projection_;
}

EnergyProjectRoll::EnergyProjectRoll(
    const std::array<double, 9>& camera_matrix,
    const std::vector<double>& dist_coeffs
):
    PnPSolver(camera_matrix, dist_coeffs) {
    pthis_ = this;
}

template<typename T>
bool EnergyProjectRoll::CostFunctor::operator()(const T* const roll, T* residual) const {
    // Caculate pose in imu
    Eigen::Matrix<T, 3, 3> rotation_mat, R_x, R_y, R_z;
    R_x << T(1), T(0), T(0), T(0), ceres::cos(*roll), -ceres::sin(*roll), T(0), ceres::sin(*roll),
        ceres::cos(*roll);
    R_y << ceres::cos(T(pthis_->pitch_)), T(0), ceres::sin(T(pthis_->pitch_)), T(0), T(1), T(0),
        -ceres::sin(T(pthis_->pitch_)), T(0), ceres::cos(T(pthis_->pitch_));
    R_z << ceres::cos(T(pthis_->yaw_)), -ceres::sin(T(pthis_->yaw_)), T(0),
        ceres::sin(T(pthis_->yaw_)), ceres::cos(T(pthis_->yaw_)), T(0), T(0), T(0), T(1);
    rotation_mat = R_z * R_y * R_x;
    // Convert pose to camera
    Eigen::Matrix<T, 3, 3> odom2cam_r;
    odom2cam_r << T(pthis_->odom2cam_r_(0, 0)), T(pthis_->odom2cam_r_(0, 1)),
        T(pthis_->odom2cam_r_(0, 2)), T(pthis_->odom2cam_r_(1, 0)), T(pthis_->odom2cam_r_(1, 1)),
        T(pthis_->odom2cam_r_(1, 2)), T(pthis_->odom2cam_r_(2, 0)), T(pthis_->odom2cam_r_(2, 1)),
        T(pthis_->odom2cam_r_(2, 2));
    rotation_mat = odom2cam_r * rotation_mat;
    // Project points
    Eigen::Matrix<T, 3, 1> tvec;
    tvec << T(pthis_->tvec_.at<double>(0, 0)), T(pthis_->tvec_.at<double>(1, 0)),
        T(pthis_->tvec_.at<double>(2, 0));
    std::vector<Eigen::Vector<T, 3>> points;
    for (const auto& point: pthis_->object_points_) {
        Eigen::Matrix<T, 3, 1> point_eigen;
        point_eigen << T(point.x), T(point.y), T(point.z);
        points.emplace_back(rotation_mat * point_eigen + tvec);
    }
    Eigen::Matrix<T, 3, 3> camera_matrix;
    camera_matrix << T(pthis_->camera_matrix_.at<double>(0, 0)),
        T(pthis_->camera_matrix_.at<double>(0, 1)), T(pthis_->camera_matrix_.at<double>(0, 2)),
        T(pthis_->camera_matrix_.at<double>(1, 0)), T(pthis_->camera_matrix_.at<double>(1, 1)),
        T(pthis_->camera_matrix_.at<double>(1, 2)), T(pthis_->camera_matrix_.at<double>(2, 0)),
        T(pthis_->camera_matrix_.at<double>(2, 1)), T(pthis_->camera_matrix_.at<double>(2, 2));
    std::vector<Eigen::Vector<T, 3>> projected_points;
    for (const auto& point: points) {
        Eigen::Vector<T, 3> projected_point;
        projected_point = camera_matrix / point[2] * point;
        projected_points.emplace_back(projected_point);
    }
    // Caculate the difference between projected points and image points (Use Vector's angle)
    Eigen::Vector<T, 3> image_vectors[2], projected_vectors[2];
    for (int i = 0; i < 2; i++) {
        image_vectors[i] << T(pthis_->image_armor_points_[i].x - pthis_->image_armor_points_[4].x),
            T(pthis_->image_armor_points_[i].y - pthis_->image_armor_points_[4].y), T(1);
        projected_vectors[i] << projected_points[i][0] - projected_points[4][0],
            projected_points[i][1] - projected_points[4][1], T(1);
    }
    Eigen::Vector<T, 3> image_vector = image_vectors[0] + image_vectors[1];
    Eigen::Vector<T, 3> projected_vector = projected_vectors[0] + projected_vectors[1];
    residual[0] = ceres::acos(
        image_vector.dot(projected_vector) / (image_vector.norm() * projected_vector.norm())
    );
    return true;
}

double EnergyProjectRoll::diff_function(double roll) {
    // Caculate rotation matrix
    cv::Mat rotation_matrix;
    get_rotation_matrix(roll, rotation_matrix);
    rotation_matrix = odom2cam_r_ * rotation_matrix;
    // Turn rotation matrix into rotation vector
    // caculate projection points
    cv::projectPoints(
        object_points_,
        rotation_matrix,
        tvec_,
        camera_matrix_,
        dist_coeffs_,
        projected_points_
    );
    // Caculate the difference between projected points and image points by angle between vectors
    double diff = 0;
    Eigen::Vector3d image_vectors[2], projected_vectors[2];
    for (int i = 0; i < 2; i++) {
        image_vectors[i] << image_armor_points_[i].x - image_armor_points_[i + 1].x,
            image_armor_points_[i].y - image_armor_points_[i + 1].y, 1;
        projected_vectors[i] << projected_points_[i].x - projected_points_[i + 1].x,
            projected_points_[i].y - projected_points_[i + 1].y, 1;
    }
    Eigen::Vector3d image_vector, projected_vector;
    image_vector = image_vectors[0] + image_vectors[1];
    projected_vector = projected_vectors[0] + projected_vectors[1];
    double diff_angle = image_vectors[0].dot(image_vectors[1])
        / (image_vectors[0].norm() * image_vectors[1].norm());
    return diff_angle;
}

void EnergyProjectRoll::get_rotation_matrix(double roll, cv::Mat& rotation_mat) const {
    // caculate rotation matrix
    // for more see paper: https://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
    // Calculate rotation about x axis
    // clang-format off
  cv::Mat R_x = (cv::Mat_<double>(3,3) <<
              1,       0,               0,
              0,       std::cos(roll), -std::sin(roll),
              0,       std::sin(roll), std::cos(roll));
    
  // Calculate rotation about y axis
  cv::Mat R_y = (cv::Mat_<double>(3,3) <<
              std::cos(pitch_),    0,      std::sin(pitch_),
              0,                   1,      0,
              -std::sin(pitch_),   0,      std::cos(pitch_));
    
  // Calculate rotation about z axis
  cv::Mat R_z = (cv::Mat_<double>(3,3) <<
              std::cos(yaw_),    -std::sin(yaw_),      0,
              std::sin(yaw_),    std::cos(yaw_),       0,
              0,                0,                   1);
    // clang-format on
    rotation_mat = R_z * R_y * R_x;
}

bool EnergyProjectRoll::solve_pose(const Armor& armor, cv::Mat& rvec, cv::Mat& tvec) {
    /*Solve PnP First To Get Position Info*/
    // Fill in image points
    if (!use_projection_) {
        return solve_pnp(armor, rvec, tvec);
    } else {
        if (!solve_pnp(armor, rvec, tvec)) {
            return false;
        }
    }
    if (!is_transform_info_updated_) {
        logger_.warn("PnP Solve done, but transform info not updated, skipping");
        return true;
    }
    /*Start Reprojection*/
    // convert armor to imu frame
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);
    cv::Mat armor_pose_in_imu = cam2odom_r_ * rotation_matrix;
    // Get yaw
    double distance = std::sqrt(
        tvec.at<double>(0, 0) * tvec.at<double>(0, 0)
        + tvec.at<double>(1, 0) * tvec.at<double>(1, 0)
        + tvec.at<double>(2, 0) * tvec.at<double>(2, 0)
    );
    // yaw_ = std::atan2(armor_pose_in_imu.at<double>(1, 0), armor_pose_in_imu.at<double>(0, 0));
    double roll = 0;
    tvec_ = tvec;
    // Get min diff roll
    ceres::Problem problem;
    ceres::CostFunction* costfunctor =
        new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(costfunctor, nullptr, &roll);
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // draw points
    diff_function(roll);
    // Caculate rotation matrix
    get_rotation_matrix(roll, rvec);
    rvec = odom2cam_r_ * rvec;
    return true;
}

void EnergyProjectRoll::draw_projection_points(cv::Mat& image) {
    if (projected_points_.empty()) {
        logger_.debug("empty projection");
        return;
    }
    // for (int i = 0; i < 5; i++) {
    //     cv::circle(image, projected_points_[i], (i + 1) * 10, cv::Scalar(0, 0, 255), -1);
    //     cv::circle(image, image_armor_points_[i], (i + 1) * 10, cv::Scalar(255, 0, 0), 2);
    // }
    cv::line(image, projected_points_[0], projected_points_[2], cv::Scalar(255, 255, 0), 2);
    cv::line(image, projected_points_[1], projected_points_[3], cv::Scalar(255, 255, 0), 2);
    cv::line(image, projected_points_[0], projected_points_[1], cv::Scalar(255, 255, 0), 2);
    cv::line(image, projected_points_[1], projected_points_[2], cv::Scalar(255, 255, 0), 2);
    cv::line(image, projected_points_[2], projected_points_[3], cv::Scalar(255, 255, 0), 2);
    cv::line(image, projected_points_[3], projected_points_[4], cv::Scalar(255, 255, 0), 2);
    cv::line(image, projected_points_[4], projected_points_[0], cv::Scalar(255, 255, 0), 2);
    projected_points_.clear();
}

void EnergyProjectRoll::update_transform_info(BaseTransformInfo* infos) {
    auto transform_infos = dynamic_cast<EnergyTransformInfo*>(infos);
    cam2odom_r_ = transform_infos->cam2odom_r.toRotMat3x3();
    odom2cam_r_ = transform_infos->odom2cam_r.toRotMat3x3();
    yaw_ = transform_infos->gimbal_yaw;
    is_transform_info_updated_ = true;
}

void EnergyProjectRoll::set_use_projection(bool use_projection) {
    use_projection_ = use_projection;
}

bool EnergyProjectRoll::use_projection() {
    return use_projection_;
}

} // namespace helios_cv