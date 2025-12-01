#!/bin/bash

# ================================================
# PCL Deprecated / Removed API Scanner
# 作者：ChatGPT，为 PCL 1.12+ 定制
# ================================================

SEARCH_DIR=${1:-"."}

echo "=============================================="
echo "   PCL Deprecated API Scanner (for PCL 1.12)"
echo "   扫描目录：$SEARCH_DIR"
echo "=============================================="
echo

# ----------- PCL 1.12 不再支持的旧 API 列表 ----------
DEPRECATED_APIS=(
    "setPolynomialFit"
    "setUpsamplingMethod"         # 需要检查枚举名
    "setUpsamplingRadius"
    "setUpsamplingStepSize"
    "setPointDensity"
    "setMLSPointDensity"
    "setDilationIterations"
    "setDilationVoxelSize"
    "setSqrGaussParam"
    "setDistinctiveNormals"
    "setPolynomialOrder"  # 有时用错类型也能报出
)

# ----------- PCL 子模块中典型移除 API ----------
DEPRECATED_APIS+=(
    "computeMLSPointNormal"
    "computeMeanAndCovarianceMatrix"
)

# ----------- 未来扩展（ROS + PCL） ----------
DEPRECATED_APIS+=(
    "ros::Time::now"      # 在 ROS2 中会变化（可忽略）
)

# ============ 扫描逻辑 =====================
for api in "${DEPRECATED_APIS[@]}"; do
    echo "🔍 正在扫描 API: $api"
    echo "-------------------------------------------"

    # 使用 grep 搜索
    grep -RIn --include=*.{cpp,h,hpp} "$api" "$SEARCH_DIR"

    if [ $? -ne 0 ]; then
        echo "✔ 未找到：$api"
    fi

    echo
done

echo "=============================================="
echo " 扫描完成。请根据输出修复 API。"
echo "=============================================="

