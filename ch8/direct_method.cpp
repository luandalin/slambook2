#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
// paths
string left_file = "./left.png";
string disparity_file = "./disparity.png";
boost::format fmt_others("./%06d.png");    // other files

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

/// class for accumulator jacobians in parallel
class JacobianAccumulator {
public:
    JacobianAccumulator(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const VecVector2d &px_ref_,
        const vector<double> depth_ref_,
        Sophus::SE3d &T21_) :
        img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    }

    /// accumulate jacobians in a range
    void accumulate_jacobian(const cv::Range &range);

    /// get hessian matrix
    Matrix6d hessian() const { return H; }

    /// get bias
    Vector6d bias() const { return b; }

    /// get total cost
    double cost_func() const { return cost; }

    /// get projected points
    VecVector2d projected_points() const { return projection; }

    /// reset h, b, cost to zero
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}

int main(int argc, char **argv) {

    cv::Mat left_img = cv::imread(left_file, 0);            //参考帧
    cv::Mat disparity_img = cv::imread(disparity_file, 0);  //视差图

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;           //边界宽度，避免采样到图像边缘区域
    VecVector2d pixels_ref;     //存储参考帧中像素坐标,每个元素是 (x, y)，位于第一帧图像坐标系
    vector<double> depth_ref;   //存储参考帧中像素对应的深度值,单位通常是米

    // generate pixels in ref and load depth data 生成参考帧像素点并读取其深度
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        
        //根据双目几何，将视差转换为深度: $depth = fx * baseline / disparity$  $Z=\frac{f_x\cdot b}{d}$
        
        int disparity = disparity_img.at<uchar>(y, x);          // 读取该像素在视差图中的视差值,使用 uchar，说明视差图是 8 位灰度
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information 使用上述的像素点和深度，估计后续多张图像的相对位姿
    Sophus::SE3d T_cur_ref;

    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }
    return 0;
}

/*
 *
 *利用参考帧像素 + 深度，通过最小化光度误差，使用 高斯–牛顿（Gauss–Newton） 在 SE(3) 上迭代估计位姿 T21 (估计当前帧相对于参考帧的位姿变换 T21)
 *
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,            //参考帧
    const cv::Mat &img2,            //当前帧
    const VecVector2d &px_ref,      //参考帧中像素坐标
    const vector<double> depth_ref,  //参考帧中像素对应的深度值
    Sophus::SE3d &T21) {

    const int iterations = 10;
    double cost = 0, lastCost = 0;
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);;
        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3) {
            // converge
            break;
        }

        lastCost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    cout << "T21 = \n" << T21.matrix() << endl;
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "direct method for single layer: " << time_used.count() << endl;

    // plot the projected pixels here
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i) {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("current", img2_show);
    cv::waitKey();
}
/*
 *
 * 对一批像素点，计算光度误差、位姿雅可比，并累加 Hessian、bias 和 cost（支持并行）
 * @param range 计算的像素点索引范围
*/
void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) { 

    // parameters
    const int half_patch_size = 1;          //使用 3x3 的小块进行光度误差计算
    int cnt_good = 0;                       //统计有效像素点数量
    Matrix6d hessian = Matrix6d::Zero();    //局部 Hessian 矩阵
    Vector6d bias = Vector6d::Zero();       //局部 bias 向量
    double cost_tmp = 0;                    //局部 cost 值

    for (size_t i = range.start; i < range.end; i++) {          //遍历该线程负责的像素点

        // compute the projection in the second image 像素 + 深度恢复 3D 点（参考帧）$P_{ref}=Z\begin{pmatrix}\frac{x-c_x}{f_x}\\\frac{y-c_y}{f_y}\\1\end{pmatrix}$
        
        Eigen::Vector3d point_ref =
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);

        // project to the second image 得到当前相机坐标系下的 3D 点 T21：参考帧 → 当前帧的位姿
        Eigen::Vector3d point_cur = T21 * point_ref;
        if (point_cur[2] < 0)   // depth invalid
            continue;
        // project to pixel coordinate 系像素坐标系下的投影点: $u=f_x\frac{X}{Z}+c_x, v=f_y\frac{Y}{Z}+c_y$
        float u = fx * point_cur[0] / point_cur[2] + cx;
        float v = fy * point_cur[1] / point_cur[2] + cy;
        //确保 patch 不越界
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
            v > img2.rows - half_patch_size)
            continue;
        //保存投影结果（用于可视化）    
        projection[i] = Eigen::Vector2d(u, v);
        //为雅可比计算做准备
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
            Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        cnt_good++;

        // ch8/直接法公式推导过程.md
        // and compute error and jacobian 遍历 3×3 patch: $x, y \in [-1, 0, +1]$

        //$q=TP\quad\quad u=\frac{1}{Z_2}Kq\tag{}$


        // $\frac{\partial e}{\partial \delta\xi} = \underbrace{\frac{\partial e}{\partial I_2}}_{-1}\cdot\underbrace{\frac{\partial I_2}{\partial u}}_{\text{图像梯度}}\cdot\underbrace{\frac{\partial u}{\partial q}}_{\text{投影雅可比}}\cdot\underbrace{\frac{\partial q}{\partial \delta\xi}}_{\text{扰动雅可比}}\tag{8.14,8.15}$        

        
        
        //$\frac { \partial u } { \partial \delta \xi } = \left[ \begin{array} { c c c c c } \frac { f _ { x } } { Z } & 0 & - \frac { f _ { x } X  } { Z ^ { 2 } }& - \frac { f _ { x } X Y } { Z ^ { 2 } } & f _ { x } + \frac { f _ { x } X ^ { 2 } } { Z ^ { 2 } } & - \frac { f _ { x } Y } { Z } \\ 0 & \frac { f _ { y } } { Z } & - \frac { f _ { y } Y } { Z ^ { 2 } } & - f _ { y } - \frac { f _ { y } Y ^ { 2 } } { Z ^ { 2 } } & \frac { f _ { y } X Y } { Z ^ { 2 } } & \frac { f _ { y } X } { Z } \end{array} \right]\tag{8.18}$    
        
        
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {

                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);
                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
                );

                // total jacobian 
                // $J=-\frac{\partial I_2}{\partial u}\cdot \frac{\partial u}{\partial \delta\xi} \tag{8.19}$
                
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
    }

    if (cnt_good) {
        // set hessian, bias and cost
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;
    }
}

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,                //参考帧，带深度信息
    const cv::Mat &img2,                //当前帧
    const VecVector2d &px_ref,          //参考帧中像素坐标
    const vector<double> depth_ref,     //参考帧中像素对应的深度值
    Sophus::SE3d &T21) {

    // parameters
    int pyramids = 4;                   //金字塔层数 4 层金字塔（第 0 层为原图，第 3 层最粗糙）
    double pyramid_scale = 0.5;         //每层尺寸为上一层的 50%（即长宽各缩一半，面积 1/4）
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids 创建图像金字塔
    vector<cv::Mat> pyr1, pyr2; // image pyramids pyr1:参考帧金字塔 pyr2:当前帧金字塔
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {                       //第 0 层为原图
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {                            //后续各层通过对上一层图像进行缩放得到 按照 pyramid_scale=0.5 比例逐层下采样
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values 备份相机内参，原因是相机内参必须随着金字塔层数变化而变化，
                                                    // 因为每层图像的尺寸不同，像素坐标系下的内参也不同，否则投影模型是错误的
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level 用于存储当前层的参考像素点
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels 缩放相机内参 $u=f_x\frac{X}{Z}+c_x,\space v=f_y\frac{Y}{Z}+c_y$
        // 如果图像尺寸缩小为原来的 s 倍，则 fx, fy, cx, cy 也要缩小为原来的 s 倍
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }

}
