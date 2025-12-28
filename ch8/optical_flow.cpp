//
// Created by Xiang on 2017/12/19.
//

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

string file_1 = "./LK1.png";  // first image
string file_2 = "./LK2.png";  // second image

/// Optical flow tracker and interface
class OpticalFlowTracker {
public:
    OpticalFlowTracker(
        const Mat &img1_,
        const Mat &img2_,
        const vector<KeyPoint> &kp1_,
        vector<KeyPoint> &kp2_,
        vector<bool> &success_,
        bool inverse_ = true, bool has_initial_ = false) :
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
        has_initial(has_initial_) {}

    void calculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false,
    bool has_initial_guess = false
);

/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false
);

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */

inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;
    
    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}

int main(int argc, char **argv) {

    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // key points, using GFTT here.在第一帧 img1 中检测 GFTT 角点（Shi-Tomasi 角点）：
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints 最多500个角点，质量水平0.01，最小距离20像素
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    // 跟踪 kp1 到 img2，得到:kp2_single：第二帧中的跟踪位置； success_single：每个关键点是否成功跟踪的标志位。
    // 未启用 inverse 模式（默认 inverse=false），也未提供初值（has_initial=false）适用于小位移场景
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);   //启用 inverse=true（使用逆向 Compositional 模式，加速且稳定）；
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by gauss-newton: " << time_used.count() << endl;

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by opencv: " << time_used.count() << endl;

    // plot the differences of those functions
    Mat img2_single;
    cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);

    return 0;
}

void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse, bool has_initial) {
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    
    /*
        parallel_for_: OpenCV 提供的并行计算接口，利用多线程加速循环任务的执行。
        作用：将一个大任务（如遍历 kp1.size() 个关键点）自动拆分成多个子任务，分配给多个 CPU 核心并行执行，显著加速计算。
        参数1：Range(0, kp1.size())
        定义任务的索引范围——从 0 到 kp1.size() - 1，每个整数索引对应一个关键点。
        参数2：std::bind(...)
        绑定一个可调用对象（函数），该函数将被每个子任务（线程）调用。
        &OpticalFlowTracker::calculateOpticalFlow：要调用的成员函数；
        &tracker：该函数作用的对象（注意：所有线程共享同一个 tracker 实例）；
        placeholders::_1：占位符，表示 parallel_for_ 会将每个子任务的 Range 子区间（如 Range(start, end)） 作为第一个参数传给 calculateOpticalFlow。
        线程安全保证：
        calculateOpticalFlow 的设计必须满足：
        只读共享数据（如 img1, img2, kp1）；
        写入位置互不重叠（如线程 A 处理 i=0~99，线程 B 处理 i=100~199，分别写 kp2[0~99] 和 kp2[100~199]）；
        无全局/静态变量竞争。
        从 OpticalFlowTracker::calculateOpticalFlow 的实现可见，它通过 Range 参数确保每个线程处理互不重叠的索引区间，因此是线程安全的。
    */
    parallel_for_(Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}
/*
 *
 * 对指定索引范围 range 内的关键点执行光流跟踪
 *
 */
void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {
    // parameters
    int half_patch_size = 4;                            //定义图像块（patch）的半宽为 4 像素 → 实际使用 8×8 的图像块（从 -4 到 +3 共 8 个像素）。
    int iterations = 10;
    for (size_t i = range.start; i < range.end; i++) {
        auto kp = kp1[i];                               //获取第 i 个关键点在第一帧（参考帧） 中的 2D 位置 kp1[i]。
        double dx = 0, dy = 0;                          // dx,dy need to be estimated 初始化光流向量 (dx, dy)，表示该点从帧1到帧2的位移。
        if (has_initial) {                              //如果 has_initial 标志为 true，则表示 kp2 已经包含了对关键点在第二帧中的初始估计位置。
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;                  //当前迭代的光度误差平方和 , 上一次迭代的误差平方和
        bool succ = true; // indicate if this point succeeded  标志位，指示该关键点的光流估计是否成功。

        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // hessian Hessian 矩阵（2×2，对应 dx, dy）
        Eigen::Vector2d b = Eigen::Vector2d::Zero();    // bias 负梯度向量（即 Jᵀ·r）
        Eigen::Vector2d J;                              // jacobian 光度误差对 (dx, dy) 的雅可比（2 维向量）
        for (int iter = 0; iter < iterations; iter++) {
            if (inverse == false) {                     //正向模式（inverse == false）：每次迭代都重新计算 H 和 b（标准 GN）
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {                                    //逆向模式（inverse == true）：H 只在第 0 次迭代计算一次，后续只更新 b（加速计算，是 SVO 等系统的核心技巧）
                // only reset b
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            // compute cost and jacobian 计算代价函数和雅可比矩阵 遍历 8×8 图像块中的每个像素（共 64 个点）。
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    //img1 中参考点灰度值 - img2 中根据当前 (dx, dy) 预测位置的灰度（需双线性插值）
                    //$e = I _ { 1 } ( x ) - I _ { 2 } ( x + d x , y + d y )$ 
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);  // Jacobian
                    //正向模式：雅可比在 第二帧 img2 上计算（随 (dx, dy) 变化）
                    //用中心差分法计算 img2 在预测点处的 x、y 方向梯度： （负号来自误差定义 $e = I_1 - I_2$） $$\frac { \partial e } { \partial \mathrm { d } } = - \nabla I _ { 2 }$$ 
   

                    //$\frac { \partial I _ { 2 } } { \partial X } \left( x ^ { \prime } , y ^ { \prime } \right) \approx \frac { I _ { 2 } \left( x ^ { \prime } + 1 , y ^ { \prime } \right) - I _ { 2 } \left( x ^ { \prime } - 1 , y ^ { \prime } \right) } { 2 }$         
                    
                    //$\frac { \partial I _ { 2 } } { \partial Y } \left( x ^ { \prime } , y ^ { \prime } \right) \approx \frac { I _ { 2 } \left( x ^ { \prime }  , y ^ { \prime }+1 \right) - I _ { 2 } \left( x ^ { \prime }  , y ^ { \prime }-1 \right) } { 2 }$
                    
                    if (inverse == false) {
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    } else if (iter == 0) {
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                   GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                   GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }

            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) {
                // converge
                break;
            }
        }

        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}

/*
 *
 * 使用多尺度图像金字塔，对 kp1 中的关键点进行由粗到精的光流跟踪，输出其在 img2 中的位置 kp2
 * 适用于大位移或快速运动场景
 *
*/
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse) {

    // parameters
    int pyramids = 4;                               //金字塔层数 4 层金字塔（第 0 层为原图，第 3 层最粗糙）
    double pyramid_scale = 0.5;                     //每层尺寸为上一层的 50%（即长宽各缩一半，面积 1/4）
    double scales[] = {1.0, 0.5, 0.25, 0.125};      //各层相对于原图的缩放比例

    // create pyramids
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {                   //第 0 层为原图，其余层通过逐层下采样生成
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "build pyramid time: " << time_used.count() << endl;

    // coarse-to-fine LK tracking in pyramids
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1) {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];          //pt.x = pt.x * 0.125; pt.y = pt.y * 0.125; 将关键点坐标缩放到最顶层金字塔尺度
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        // from coarse to fine
        success.clear();
        t1 = chrono::steady_clock::now();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "track pyr " << level << " cost time: " << time_used.count() << endl;

        if (level > 0) {
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;         //将关键点坐标放大到下一层金字塔尺度:kp.pt = kp.pt / 0.5 =>  kp.pt = kp.pt * 2.0
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }

    for (auto &kp: kp2_pyr)
        kp2.push_back(kp);
}
