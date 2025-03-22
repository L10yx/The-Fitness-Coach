<template>
  <view class="stats-container">
    <!-- 顶部导航栏 -->
    <view class="nav-bar">
      <text class="page-title">统计</text>
    </view>
    
    <!-- 内容区域 -->
    <scroll-view scroll-y class="stats-content">
      <!-- 数据概览卡片 -->
      <view class="stats-card overview-card">
        <text class="card-title">本周概览</text>
        <view class="overview-grid">
          <view class="overview-item">
            <text class="overview-value">{{ weeklyStats.totalWorkouts }}</text>
            <text class="overview-label">训练次数</text>
          </view>
          <view class="overview-item">
            <text class="overview-value">{{ weeklyStats.totalExercises }}</text>
            <text class="overview-label">总运动量</text>
          </view>
          <view class="overview-item">
            <text class="overview-value">{{ weeklyStats.totalMinutes }}分钟</text>
            <text class="overview-label">训练时长</text>
          </view>
          <view class="overview-item">
            <text class="overview-value">{{ weeklyStats.streak }}天</text>
            <text class="overview-label">连续训练</text>
          </view>
        </view>
      </view>
      
      <!-- 训练频率图表 -->
      <view class="stats-card">
        <text class="card-title">训练频率</text>
        <view class="chart-container frequency-chart">
          <view 
            v-for="(day, index) in weeklyFrequency" 
            :key="index"
            class="frequency-bar-container"
          >
            <view 
              class="frequency-bar" 
              :style="{ height: day.height + '%' }"
              :class="{ 'active-day': day.isToday }"
            ></view>
            <text class="frequency-label">{{ day.label }}</text>
          </view>
        </view>
      </view>
      
      <!-- 运动分布图表 -->
      <view class="stats-card">
        <text class="card-title">运动分布</text>
        <view class="chart-container distribution-chart">
          <view 
            v-for="(exercise, index) in exerciseDistribution" 
            :key="index"
            class="bubble"
            :style="{
              width: exercise.size + 'rpx',
              height: exercise.size + 'rpx',
              backgroundColor: exercise.color,
              left: exercise.x + '%',
              top: exercise.y + '%'
            }"
          >
            <text class="bubble-text">{{ exercise.name }}</text>
          </view>
        </view>
      </view>
      
      <!-- 进度趋势图表 -->
      <view class="stats-card">
        <text class="card-title">进度趋势</text>
        <view class="chart-tabs">
          <text 
            v-for="(tab, index) in progressTabs" 
            :key="index"
            :class="['tab-item', activeProgressTab === index ? 'active-tab' : '']"
            @click="activeProgressTab = index"
          >{{ tab }}</text>
        </view>
        <view class="chart-container progress-chart">
          <view class="chart-legend">
            <view class="legend-item">
              <view class="legend-color" style="background-color: #FF4785;"></view>
              <text class="legend-text">引体向上</text>
            </view>
            <view class="legend-item">
              <view class="legend-color" style="background-color: #4ECDC4;"></view>
              <text class="legend-text">俯卧撑</text>
            </view>
            <view class="legend-item">
              <view class="legend-color" style="background-color: #FFD166;"></view>
              <text class="legend-text">卷腹</text>
            </view>
          </view>
          <view class="line-chart">
            <view 
              v-for="(point, pIndex) in progressData[activeProgressTab].pullup" 
              :key="'pullup-' + pIndex"
              class="chart-point"
              :style="{
                left: (pIndex / (progressData[activeProgressTab].pullup.length - 1)) * 100 + '%',
                bottom: point + '%',
                backgroundColor: '#FF4785'
              }"
            ></view>
            <view 
              v-for="(i, index) in progressData[activeProgressTab].pullup.length - 1" 
              :key="'pullup-line-' + index"
              class="chart-line"
              :style="{
                left: (index / (progressData[activeProgressTab].pullup.length - 1)) * 100 + '%',
                width: (1 / (progressData[activeProgressTab].pullup.length - 1)) * 100 + '%',
                bottom: progressData[activeProgressTab].pullup[index] + '%',
                height: Math.abs(progressData[activeProgressTab].pullup[index + 1] - progressData[activeProgressTab].pullup[index]) + '%',
                backgroundColor: '#FF4785',
                transform: progressData[activeProgressTab].pullup[index] < progressData[activeProgressTab].pullup[index + 1] ? 'none' : 'scaleY(-1)'
              }"
            ></view>
            
            <!-- 俯卧撑数据点和连线 -->
            <view 
              v-for="(point, pIndex) in progressData[activeProgressTab].pushup" 
              :key="'pushup-' + pIndex"
              class="chart-point"
              :style="{
                left: (pIndex / (progressData[activeProgressTab].pushup.length - 1)) * 100 + '%',
                bottom: point + '%',
                backgroundColor: '#4ECDC4'
              }"
            ></view>
            <view 
              v-for="(i, index) in progressData[activeProgressTab].pushup.length - 1" 
              :key="'pushup-line-' + index"
              class="chart-line"
              :style="{
                left: (index / (progressData[activeProgressTab].pushup.length - 1)) * 100 + '%',
                width: (1 / (progressData[activeProgressTab].pushup.length - 1)) * 100 + '%',
                bottom: progressData[activeProgressTab].pushup[index] + '%',
                height: Math.abs(progressData[activeProgressTab].pushup[index + 1] - progressData[activeProgressTab].pushup[index]) + '%',
                backgroundColor: '#4ECDC4',
                transform: progressData[activeProgressTab].pushup[index] < progressData[activeProgressTab].pushup[index + 1] ? 'none' : 'scaleY(-1)'
              }"
            ></view>
            
            <!-- 卷腹数据点和连线 -->
            <view 
              v-for="(point, pIndex) in progressData[activeProgressTab].crunch" 
              :key="'crunch-' + pIndex"
              class="chart-point"
              :style="{
                left: (pIndex / (progressData[activeProgressTab].crunch.length - 1)) * 100 + '%',
                bottom: point + '%',
                backgroundColor: '#FFD166'
              }"
            ></view>
            <view 
              v-for="(i, index) in progressData[activeProgressTab].crunch.length - 1" 
              :key="'crunch-line-' + index"
              class="chart-line"
              :style="{
                left: (index / (progressData[activeProgressTab].crunch.length - 1)) * 100 + '%',
                width: (1 / (progressData[activeProgressTab].crunch.length - 1)) * 100 + '%',
                bottom: progressData[activeProgressTab].crunch[index] + '%',
                height: Math.abs(progressData[activeProgressTab].crunch[index + 1] - progressData[activeProgressTab].crunch[index]) + '%',
                backgroundColor: '#FFD166',
                transform: progressData[activeProgressTab].crunch[index] < progressData[activeProgressTab].crunch[index + 1] ? 'none' : 'scaleY(-1)'
              }"
            ></view>
            
            <!-- X轴标签 -->
            <view class="x-axis-labels">
              <text 
                v-for="(label, lIndex) in progressData[activeProgressTab].labels" 
                :key="lIndex"
                class="axis-label"
                :style="{
                  left: (lIndex / (progressData[activeProgressTab].labels.length - 1)) * 100 + '%',
                }"
              >{{ label }}</text>
            </view>
          </view>
        </view>
      </view>
      
      <!-- 训练建议 -->
      <view class="stats-card">
        <text class="card-title">训练建议</text>
        <view class="advice-container">
          <view class="advice-item" v-for="(advice, index) in trainingAdvice" :key="index">
            <view class="advice-icon" :style="{ backgroundColor: advice.color }">
              <text class="icon-text">{{ advice.icon }}</text>
            </view>
            <view class="advice-content">
              <text class="advice-title">{{ advice.title }}</text>
              <text class="advice-text">{{ advice.text }}</text>
            </view>
          </view>
        </view>
      </view>
    </scroll-view>
    
    <!-- 底部导航栏 -->
    <view class="tab-bar">
      <view class="tab-item" @click="goToDiscover">
        <text class="tab-icon">⚪</text>
        <text class="tab-text">发现</text>
      </view>
      <view class="tab-item" @click="goToMain">
        <text class="tab-icon">⚪</text>
        <text class="tab-text">开始</text>
      </view>
      <view class="tab-item active">
        <text class="tab-icon">⚪</text>
        <text class="tab-text">统计</text>
      </view>
    </view>
  </view>
</template>

<script>
import userManager from '@/utils/userManager.js';

export default {
  data() {
    return {
      // 本周概览数据
      weeklyStats: {
        totalWorkouts: 5,
        totalExercises: 120,
        totalMinutes: 75,
        streak: 3
      },
      
      // 每周训练频率数据
      weeklyFrequency: [
        { label: '一', height: 30, isToday: false },
        { label: '二', height: 45, isToday: false },
        { label: '三', height: 20, isToday: false },
        { label: '四', height: 60, isToday: false },
        { label: '五', height: 40, isToday: true },
        { label: '六', height: 0, isToday: false },
        { label: '日', height: 0, isToday: false }
      ],
      
      // 运动分布数据
      exerciseDistribution: [
        { name: '引体向上', size: 160, color: 'rgba(255, 71, 133, 0.7)', x: 30, y: 30 },
        { name: '俯卧撑', size: 200, color: 'rgba(78, 205, 196, 0.7)', x: 65, y: 60 },
        { name: '卷腹', size: 120, color: 'rgba(255, 209, 102, 0.7)', x: 20, y: 70 },
        { name: '深蹲', size: 100, color: 'rgba(161, 134, 190, 0.7)', x: 70, y: 20 }
      ],
      
      // 进度趋势选项卡
      progressTabs: ['周', '月', '年'],
      activeProgressTab: 0,
      
      // 进度趋势数据
      progressData: [
        { // 周数据
          pullup: [10, 15, 20, 25, 20, 30, 35],
          pushup: [30, 35, 40, 45, 50, 55, 60],
          crunch: [20, 25, 30, 35, 40, 35, 45],
          labels: ['一', '二', '三', '四', '五', '六', '日']
        },
        { // 月数据
          pullup: [15, 20, 25, 30],
          pushup: [35, 45, 50, 60],
          crunch: [25, 30, 40, 45],
          labels: ['第一周', '第二周', '第三周', '第四周']
        },
        { // 年数据
          pullup: [10, 15, 25, 30, 35, 40],
          pushup: [30, 40, 45, 55, 65, 70],
          crunch: [20, 25, 35, 40, 45, 50],
          labels: ['1月', '3月', '5月', '7月', '9月', '11月']
        }
      ],
      
      // 训练建议
      trainingAdvice: [
        {
          icon: '↑',
          color: '#FF4785',
          title: '增加引体向上训练频率',
          text: '建议每周至少训练3次引体向上，可以提高背部和手臂肌群。'
        },
        {
          icon: '↗',
          color: '#4ECDC4',
          title: '俯卧撑进步明显',
          text: '继续保持当前训练强度，可以尝试增加组数或变换姿势提高难度。'
        },
        {
          icon: '!',
          color: '#FFD166',
          title: '注意训练平衡',
          text: '您的上肢训练较多，建议增加下肢训练如深蹲和弓步蹲，保持全身肌肉平衡发展。'
        }
      ]
    }
  },
  methods: {
    goToDiscover() {
      uni.redirectTo({
        url: '/pages/discover/discover'
      });
    },
    goToMain() {
      uni.redirectTo({
        url: '/pages/main/main'
      });
    },
    // 加载用户数据的方法，实际应用中可以从userManager获取
    loadUserData() {
      // 这里可以添加从userManager获取用户训练数据的逻辑
      // 例如：this.weeklyStats = userManager.getUserWeeklyStats();
      console.log('加载用户数据');
    }
  },
  onShow() {
    // 每次页面显示时加载最新数据
    this.loadUserData();
  }
}
</script>

<style>
.stats-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: #121212;
  color: #ffffff;
  padding: 40rpx;
  box-sizing: border-box;
  position: relative;
  padding-bottom: 140rpx; /* 为底部导航栏留出空间 */
}

/* 顶部导航栏样式 */
.nav-bar {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 90rpx;
  margin-bottom: 30rpx;
  position: relative;
}

.page-title {
  font-size: 36rpx;
  font-weight: bold;
  color: #ffffff;
}

/* 内容区域样式 */
.stats-content {
  flex: 1;
  height: calc(100vh - 230rpx);
}

/* 卡片通用样式 */
.stats-card {
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 20rpx;
  padding: 30rpx;
  margin-bottom: 30rpx;
}

.card-title {
  font-size: 32rpx;
  font-weight: bold;
  color: #ffffff;
  margin-bottom: 20rpx;
  display: block;
}

/* 概览卡片样式 */
.overview-grid {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
}

.overview-item {
  width: 48%;
  background-color: rgba(255, 255, 255, 0.03);
  border-radius: 16rpx;
  padding: 20rpx;
  margin-bottom: 20rpx;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.overview-value {
  font-size: 40rpx;
  font-weight: bold;
  color: #FF4785;
  margin-bottom: 10rpx;
}

.overview-label {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

/* 频率图表样式 */
.chart-container {
  width: 100%;
  height: 300rpx;
  position: relative;
  margin-top: 20rpx;
}

.frequency-chart {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  height: 250rpx;
}

.frequency-bar-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 12%;
}

.frequency-bar {
  width: 100%;
  background: linear-gradient(180deg, #FF4785 0%, #FF8D4E 100%);
  border-radius: 8rpx 8rpx 0 0;
  transition: height 0.3s ease;
}

.active-day {
  background: linear-gradient(180deg, #4ECDC4 0%, #556270 100%);
}

.frequency-label {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
  margin-top: 10rpx;
}

/* 分布图表样式 */
.distribution-chart {
  height: 400rpx;
  position: relative;
}

.bubble {
  position: absolute;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transform: translate(-50%, -50%);
}

.bubble-text {
  font-size: 24rpx;
  color: #ffffff;
  font-weight: bold;
  text-align: center;
}

/* 进度趋势图表样式 */
.chart-tabs {
  display: flex;
  margin-bottom: 20rpx;
}

.tab-item {
  padding: 10rpx 30rpx;
  font-size: 28rpx;
  color: rgba(255, 255, 255, 0.6);
  border-radius: 30rpx;
  margin-right: 20rpx;
}

.active-tab {
  background-color: rgba(255, 71, 133, 0.2);
  color: #FF4785;
}

.chart-legend {
  display: flex;
  margin-bottom: 20rpx;
}

.legend-item {
  display: flex;
  align-items: center;
  margin-right: 30rpx;
}

.legend-color {
  width: 20rpx;
  height: 20rpx;
  border-radius: 10rpx;
  margin-right: 10rpx;
}

.legend-text {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

.line-chart {
  height: 200rpx;
  position: relative;
  margin-top: 30rpx;
}

.chart-point {
  width: 12rpx;
  height: 12rpx;
  border-radius: 6rpx;
  position: absolute;
  transform: translate(-50%, 50%);
}

.chart-line {
  position: absolute;
  transform-origin: bottom left;
}

.x-axis-labels {
  position: absolute;
  bottom: -30rpx;
  left: 0;
  right: 0;
  display: flex;
}

.axis-label {
  position: absolute;
  font-size: 22rpx;
  color: rgba(255, 255, 255, 0.5);
  transform: translateX(-50%);
}

/* 训练建议样式 */
.advice-container {
  display: flex;
  flex-direction: column;
}

.advice-item {
  display: flex;
  align-items: flex-start;
  margin-bottom: 20rpx;
  background-color: rgba(255, 255, 255, 0.03);
  border-radius: 16rpx;
  padding: 20rpx;
}

.advice-icon {
  width: 60rpx;
  height: 60rpx;
  border-radius: 30rpx;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 20rpx;
  flex-shrink: 0;
}

.icon-text {
  font-size: 28rpx;
  color: #ffffff;
  font-weight: bold;
}

.advice-content {
  flex: 1;
}

.advice-title {
  font-size: 28rpx;
  font-weight: bold;
  color: #ffffff;
  margin-bottom: 10rpx;
}

.advice-text {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
  line-height: 1.4;
}

/* 底部导航栏样式 */
.tab-bar {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  height: 120rpx;
  background-color: rgba(18, 18, 18, 0.95);
  display: flex;
  justify-content: space-around;
  align-items: center;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  padding-bottom: env(safe-area-inset-bottom);
}

.tab-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 33.33%;
  height: 100%;
}

.tab-icon {
  font-size: 44rpx;
  color: rgba(255, 255, 255, 0.6);
  margin-bottom: 8rpx;
  transition: color 0.3s ease;
}

.tab-text {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

.tab-item.active .tab-icon,
.tab-item.active .tab-text {
  color: #FF4785;
}
</style>