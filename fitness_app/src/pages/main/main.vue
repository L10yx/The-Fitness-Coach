<template>
  <view class="main-container">
    <!-- 顶部用户信息栏 -->
    <view class="user-info-bar">
      <view class="user-profile" @click="goToProfile">
        <view class="avatar-container">
          <text class="avatar-placeholder">用户</text>
        </view>
        <view class="user-details">
          <text class="username">健身哥</text>
          <text class="user-level">Level 1</text>
        </view>
      </view>
      <view class="settings-icon" @click="goToSettings">
        <text class="icon-text">设置</text>
      </view>
    </view>
    
    <!-- 欢迎信息 -->
    <view class="welcome-section">
      <text class="welcome-text">今天想要挑战什么运动？</text>
      <text class="welcome-subtext">AI将为您提供精准计数</text>
    </view>
    
    <!-- 运动选择区域 -->
    <view class="exercises-section">
      <text class="section-title">选择运动</text>
      
      <exercise-card 
        name="引体向上" 
        description="锻炼背部和手臂肌群"
        icon-text="引体"
        exercise-type="pullup"
        @exercise-selected="handleExerciseSelected"
      />
      
      <exercise-card 
        name="俯卧撑" 
        description="增强上肢和胸部力量"
        icon-text="俯卧"
        exercise-type="pushup"
        @exercise-selected="handleExerciseSelected"
      />
      
      <exercise-card 
        name="卷腹" 
        description="塑造腹肌线条"
        icon-text="卷腹"
        exercise-type="crunch"
        @exercise-selected="handleExerciseSelected"
      />
    </view>
    
    <!-- 历史记录快捷入口 -->
    <view class="history-shortcut" @click="goToHistory">
      <text class="history-text">查看历史记录</text>
      <text class="history-icon">→</text>
    </view>
    
    <!-- 底部导航栏 -->
    <view class="tab-bar">
      <view class="tab-item" @click="goToDiscover">
        <text class="tab-icon">⚪</text>
        <text class="tab-text">发现</text>
      </view>
      <view class="tab-item active">
        <text class="tab-icon">⚪</text>
        <text class="tab-text">开始</text>
      </view>
      <view class="tab-item" @click="goToStats">
        <text class="tab-icon">⚪</text>
        <text class="tab-text">统计</text>
      </view>
    </view>
  </view>
</template>

<script>
import ExerciseCard from '@/components/ExerciseCard.vue';

export default {
  components: {
    ExerciseCard
  },
  data() {
    return {
      // 数据可以根据需要扩展
    }
  },
  methods: {
    handleExerciseSelected(exerciseType) {
      console.log('选择的运动:', exerciseType);
      // 导航到运动计数页面
      uni.navigateTo({
        url: `/pages/exercise/exercise?type=${exerciseType}`
      });
    },
    goToSettings() {
      uni.navigateTo({
        url: '/pages/settings/settings'
      });
    },
    goToHistory() {
      uni.navigateTo({
        url: '/pages/history/history'
      });
    },
    goToStats() {
      uni.redirectTo({
        url: '/pages/stats/stats'
      });
    },
    goToProfile() {
      uni.navigateTo({
        url: '/pages/profile/profile'
      });
    },
    goToDiscover() {
      uni.redirectTo({
        url: '/pages/discover/discover'
      });
    }
  }
}
</script>

<style>
.main-container {
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

/* 用户信息栏样式 */
.user-info-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 40rpx;
}

.user-profile {
  display: flex;
  align-items: center;
}

.avatar-container {
  width: 80rpx;
  height: 80rpx;
  border-radius: 40rpx;
  background: linear-gradient(135deg, #FF4785 0%, #FF8D4E 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 20rpx;
}

.avatar-placeholder {
  font-size: 28rpx;
  color: #ffffff;
  font-weight: bold;
}

.user-details {
  display: flex;
  flex-direction: column;
}

.username {
  font-size: 32rpx;
  font-weight: bold;
  color: #ffffff;
}

.user-level {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
}

.settings-icon {
  width: 70rpx;
  height: 70rpx;
  border-radius: 35rpx;
  background-color: rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
}

.icon-text {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.8);
}

/* 欢迎区域样式 */
.welcome-section {
  margin-bottom: 50rpx;
  display: flex;
  flex-direction: column;
}

.welcome-text {
  font-size: 44rpx;
  font-weight: bold;
  color: #ffffff;
  margin-bottom: 16rpx;
}

.welcome-subtext {
  font-size: 28rpx;
  color: rgba(255, 255, 255, 0.6);
  line-height: 1.4;
}

/* 运动选择区域样式 */
.exercises-section {
  margin-bottom: 40rpx;
}

.section-title {
  font-size: 36rpx;
  font-weight: bold;
  color: #ffffff;
  margin-bottom: 30rpx;
  display: block;
}

/* 历史记录快捷入口 */
.history-shortcut {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 16rpx;
  padding: 30rpx;
  margin-top: 20rpx;
}

.history-text {
  font-size: 30rpx;
  color: #ffffff;
}

.history-icon {
  font-size: 30rpx;
  color: #FF4785;
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