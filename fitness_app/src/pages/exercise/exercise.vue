<template>
  <view class="exercise-container">
    <!-- 顶部信息栏 -->
    <view class="top-bar">
      <view class="back-button" @click="goBack">
        <text class="back-icon">←</text>
      </view>
      <view class="exercise-title">
        <text class="title-text">{{ exerciseTitle }}</text>
      </view>
      <view class="placeholder"></view>
    </view>
    
    <!-- 摄像头区域 -->
    <view class="camera-container">
      <view v-if="!cameraActive" class="camera-placeholder">
        <text class="placeholder-text">点击开始按钮启动摄像头</text>
      </view>
      <camera v-else device-position="front" flash="off" class="camera" @error="handleCameraError"></camera>
      
      <!-- 计数信息覆盖层 -->
      <view class="counter-overlay">
        <view class="counter-box">
          <text class="counter-number">{{ counter }}</text>
          <text class="counter-label">次数</text>
        </view>
      </view>
    </view>
    
    <!-- 运动信息区域 -->
    <view class="exercise-info">
      <view class="info-item">
        <text class="info-label">运动项目</text>
        <text class="info-value">{{ exerciseTitle }}</text>
      </view>
      <view class="info-item">
        <text class="info-label">运动时间</text>
        <text class="info-value">{{ formatTime(elapsedTime) }}</text>
      </view>
      <view class="info-item">
        <text class="info-label">完成数量</text>
        <text class="info-value">{{ counter }}</text>
      </view>
    </view>
    
    <!-- 控制按钮区域 -->
    <view class="control-buttons">
      <view v-if="!exerciseStarted" class="start-button" @click="startExercise">
        <text class="button-text">开始</text>
      </view>
      <view v-else class="button-group">
        <view :class="['control-button', isPaused ? 'resume-button' : 'pause-button']" @click="togglePause">
          <text class="button-text">{{ isPaused ? '继续' : '暂停' }}</text>
        </view>
        <view class="control-button end-button" @click="endExercise">
          <text class="button-text">结束</text>
        </view>
      </view>
    </view>
  </view>
</template>

<script>
import userManager from '@/utils/userManager.js';

export default {
  data() {
    return {
      exerciseType: '',
      exerciseTitle: '',
      counter: 0,
      elapsedTime: 0,
      timer: null,
      exerciseStarted: false,
      isPaused: false,
      cameraActive: false,
      exerciseTitles: {
        pullup: '引体向上',
        pushup: '俯卧撑',
        crunch: '卷腹'
      }
    }
  },
  onLoad(options) {
    // 获取从主页传递的运动类型参数
    if (options.type) {
      this.exerciseType = options.type;
      this.exerciseTitle = this.exerciseTitles[this.exerciseType] || '未知运动';
    }
  },
  onUnload() {
    // 页面卸载时清除计时器
    this.clearTimer();
  },
  methods: {
    startExercise() {
      this.exerciseStarted = true;
      this.isPaused = false;
      this.cameraActive = true;
      
      // 启动计时器
      this.startTimer();
      
      // 这里可以添加AI计数的初始化逻辑
      uni.showToast({
        title: `${this.exerciseTitle}计数开始`,
        icon: 'none'
      });
    },
    togglePause() {
      this.isPaused = !this.isPaused;
      
      if (this.isPaused) {
        // 暂停计时器
        this.clearTimer();
        uni.showToast({
          title: '已暂停',
          icon: 'none'
        });
      } else {
        // 继续计时器
        this.startTimer();
        uni.showToast({
          title: '已继续',
          icon: 'none'
        });
      }
    },
    endExercise() {
      // 显示确认对话框
      uni.showModal({
        title: '结束运动',
        content: '确定要结束当前运动吗？',
        success: (res) => {
          if (res.confirm) {
            // 停止计时器
            this.clearTimer();
            this.cameraActive = false;
            
            // 保存运动记录
            this.saveExerciseRecord();
            
            // 显示结果页面或返回主页
            this.showResult();
          }
        }
      });
    },
    startTimer() {
      // 清除可能存在的计时器
      this.clearTimer();
      
      // 启动新的计时器，每秒更新一次
      this.timer = setInterval(() => {
        this.elapsedTime++;
      }, 1000);
    },
    clearTimer() {
      if (this.timer) {
        clearInterval(this.timer);
        this.timer = null;
      }
    },
    formatTime(seconds) {
      // 将秒数格式化为 mm:ss 格式
      const mins = Math.floor(seconds / 60);
      const secs = seconds % 60;
      return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    },
    saveExerciseRecord() {
      // 保存运动记录到用户账号下
      const record = {
        type: this.exerciseType,
        title: this.exerciseTitle,
        count: this.counter,
        duration: this.elapsedTime,
        date: new Date().toISOString()
      };
      
      // 使用userManager添加记录
      const success = userManager.addExerciseRecord(record);
      
      if (!success) {
        console.warn('保存运动记录失败，可能未登录');
        // 如果未登录，可以提示用户或使用本地存储作为备份
        uni.showToast({
          title: '请登录以保存记录',
          icon: 'none'
        });
      } else {
        console.log('成功保存运动记录:', record);
      }
    },
    showResult() {
      // 显示运动结果或返回主页
      // 使用redirectTo而不是navigateBack，避免页面堆栈问题导致的闪烁
      uni.showToast({
        title: `完成${this.counter}个${this.exerciseTitle}`,
        icon: 'success',
        duration: 2000
      });
      
      // 不在Toast的回调中执行，而是直接设置定时器，避免回调嵌套可能导致的闪烁
      setTimeout(() => {
        // 使用redirectTo替代navigateBack，避免页面堆栈问题
        uni.redirectTo({
          url: '/pages/main/main'
        });
      }, 2500); // 稍微延长时间，确保Toast显示完全
    },
    goBack() {
      // 如果运动已开始且未结束，显示确认对话框
      if (this.exerciseStarted && this.cameraActive) {
        uni.showModal({
          title: '确认返回',
          content: '运动尚未结束，确定要返回吗？',
          success: (res) => {
            if (res.confirm) {
              uni.navigateBack();
            }
          }
        });
      } else {
        uni.navigateBack();
      }
    },
    handleCameraError(e) {
      console.error('相机错误:', e);
      uni.showToast({
        title: '相机启动失败，请检查权限设置',
        icon: 'none'
      });
      this.cameraActive = false;
    },
    // 模拟AI计数增加的方法（实际项目中应由AI视觉识别触发）
    // 这里仅作为演示，实际应用中应删除此方法
    simulateCount() {
      // 仅在运动已开始且未暂停时增加计数
      if (this.exerciseStarted && !this.isPaused && this.cameraActive) {
        this.counter++;
      }
    }
  }
}
</script>

<style>
.exercise-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: #121212;
  color: #ffffff;
  position: relative;
}

/* 顶部栏样式 */
.top-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 40rpx 30rpx;
  background-color: rgba(0, 0, 0, 0.3);
}

.back-button {
  width: 70rpx;
  height: 70rpx;
  border-radius: 35rpx;
  background-color: rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
}

.back-icon {
  font-size: 36rpx;
  color: #ffffff;
}

.exercise-title {
  font-size: 36rpx;
  font-weight: bold;
}

.placeholder {
  width: 70rpx;
}

/* 摄像头区域样式 */
.camera-container {
  width: 100%;
  height: 750rpx;
  position: relative;
  background-color: #1a1a1a;
  overflow: hidden;
}

.camera {
  width: 100%;
  height: 100%;
}

.camera-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #1a1a1a;
}

.placeholder-text {
  font-size: 32rpx;
  color: rgba(255, 255, 255, 0.6);
}

/* 计数覆盖层样式 */
.counter-overlay {
  position: absolute;
  top: 20rpx;
  right: 20rpx;
  z-index: 10;
}

.counter-box {
  background: linear-gradient(135deg, rgba(255, 71, 133, 0.8) 0%, rgba(255, 141, 78, 0.8) 100%);
  border-radius: 20rpx;
  padding: 20rpx;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.counter-number {
  font-size: 60rpx;
  font-weight: bold;
  color: #ffffff;
}

.counter-label {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.8);
}

/* 运动信息区域样式 */
.exercise-info {
  display: flex;
  justify-content: space-between;
  padding: 40rpx;
  background-color: rgba(255, 255, 255, 0.05);
}

.info-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.info-label {
  font-size: 24rpx;
  color: rgba(255, 255, 255, 0.6);
  margin-bottom: 10rpx;
}

.info-value {
  font-size: 36rpx;
  font-weight: bold;
  color: #ffffff;
}

/* 控制按钮区域样式 */
.control-buttons {
  padding: 40rpx;
  display: flex;
  justify-content: center;
  margin-top: auto;
  margin-bottom: 40rpx;
}

.start-button {
  width: 400rpx;
  height: 100rpx;
  background: linear-gradient(90deg, #FF4785 0%, #FF8D4E 100%);
  border-radius: 50rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

.button-group {
  display: flex;
  width: 100%;
  justify-content: space-between;
}

.control-button {
  width: 300rpx;
  height: 100rpx;
  border-radius: 50rpx;
  display: flex;
  align-items: center;
  justify-content: center;
}

.pause-button {
  background-color: rgba(255, 255, 255, 0.2);
}

.resume-button {
  background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
}

.end-button {
  background: linear-gradient(90deg, #F44336 0%, #FF9800 100%);
}

.button-text {
  font-size: 32rpx;
  font-weight: bold;
  color: #ffffff;
}
</style>