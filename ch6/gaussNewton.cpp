#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
  //$$J ( x ) ^ { T } J ( x ) \Delta x = - J ( x ) ^ { T } f ( x ) .$$
  

  //$$\sum_{n=0}^{100}J ( x )_n ^ { T } J ( x )_n\left[\begin{array} { l } \Delta a\\\Delta b\\\Delta c\end{array}\right]=\sum_{n=0}^{100}\left(-J(x)_nf(x)_n\right)$$

  
  //$$H \Delta x = b $$
  
int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
  //高斯牛顿第一步：给定初值$x_0$
  double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
  int N = 100;                                 // 数据点
  double w_sigma = 1.0;                        // 噪声Sigma值
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;                                 // OpenCV随机数产生器

  //生成带噪声的数据曲线
  vector<double> x_data, y_data;      // 数据
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
  }

  // 开始Gauss-Newton迭代
  int iterations = 100;    // 迭代次数
  double cost = 0, lastCost = 0;  // 本次迭代的cost和上一次迭代的cost

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for (int iter = 0; iter < iterations; iter++) {

    Matrix3d H = Matrix3d::Zero();             // $$Hessian = J^T W^{-1} J$$ in Gauss-Newton
    Vector3d b = Vector3d::Zero();             // bias
    cost = 0;

    for (int i = 0; i < N; i++) {
      double xi = x_data[i], yi = y_data[i];  // 第i个数据点
      //高斯牛顿第二步：对于第 k 次迭代，求出当前的雅可比矩阵 $J (x_k )$ 和误差 $f (x_k )$。
      //$$\textcolor{red}{error=y-\exp(ax^2+bx+c)}$$ 
      double error = yi - exp(ae * xi * xi + be * xi + ce);//$$error = y-f(x)$$
      Vector3d J; // 雅可比矩阵
      J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da $$\frac{\partial e}{\partial a}=\frac{\partial(y-\exp(ax^2+bx+c))}{\partial a}=-x^2\cdot \exp(ax^2+bx+c)$$
      J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
      J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc
      //$J^\top J$可能奇异或病态
      H += inv_sigma * inv_sigma * J * J.transpose();  //$H=J\cdot W^{-1}\cdot J^\top$
      b += -inv_sigma * inv_sigma * error * J;

      cost += error * error;
    }

    // 求解线性方程 Hx=b
    //高斯牛顿第三步：解线性方程以获得参数更新量 $\Delta x$.
    Vector3d dx = H.ldlt().solve(b);  //求解$\Delta x$
    if (isnan(dx[0])) {
      cout << "result is nan!" << endl;
      break;
    }
    //高斯牛顿第四步：若 cost 大于等于last cost，则停止。否则，令 x k+1 = x k + ∆x k ，否则返回 第二步.
    if (iter > 0 && cost >= lastCost) {
      cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
      break;
    }
    //$$x_{k+1} = x_k + \Delta x_k$$
    ae += dx[0];
    be += dx[1];
    ce += dx[2];

    lastCost = cost;

    cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
         "\t\testimated params: " << ae << "," << be << "," << ce << endl;
  }

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
  return 0;
}
